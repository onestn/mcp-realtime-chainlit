import io
import os
import base64
import aiohttp
import asyncio
import mimetypes
import traceback

from PIL import Image
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List
from openai import AsyncOpenAI

# chainlit
import chainlit as cl
from chainlit.logger import logger


load_dotenv(override=True)


def _compose_guidance() -> str:
    input_path = os.environ.get("PROMPT_INPUT_PATH", "prompts/input.md")
    output_path = os.environ.get("PROMPT_OUTPUT_PATH", "prompts/output.md")
    input_text = Path(input_path).read_text(encoding="utf-8").strip()
    output_text = Path(output_path).read_text(encoding="utf-8").strip()
    parts = []
    if input_text:
        parts.append(f"[입력 지침]\n{input_text}")
    if output_text:
        parts.append(f"[출력 지침]\n{output_text}")
    return "\n\n".join(parts).strip()


async def _read_file_bytes(file) -> bytes:
    if hasattr(file, "content") and file.content:
        if isinstance(file.content, (bytes, bytearray)):
            return bytes(file.content)
        if hasattr(file.content, "read"):
            return file.content.read()

    file_path = getattr(file, "path", None)
    if file_path and os.path.exists(file_path):
        return await asyncio.to_thread(Path(file_path).read_bytes)

    file_url = getattr(file, "url", None)
    if file_url:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                resp.raise_for_status()
                return await resp.read()

    raise ValueError("이미지 데이터를 읽을 수 없습니다.")


def _extract_text_from_response(response: Dict[str, Any]) -> str:

    outputs: List[Dict[str, Any]] = response.get("output", []) or []
    if not outputs:
        outputs = response.get("choices", [])  # fallback shape

    collected: List[str] = []
    for item in outputs:
        contents = item.get("content", []) if isinstance(item, dict) else []
        for part in contents:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"output_text", "text"}:
                collected.append(part.get("text", ""))
            elif part_type == "message":
                nested_content = part.get("content", [])
                for nested in nested_content:
                    if nested.get("type") in {"output_text", "text"}:
                        collected.append(nested.get("text", ""))

    text = "\n".join(filter(None, (segment.strip() for segment in collected)))
    return text.strip()


async def _analyze_image_with_vision_api(prompt: str, image_data_url: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VISION_API_KEY")
    model = (
        os.environ.get("VISION_MODEL")
        or os.environ.get("OPENAI_VISION_MODEL")
        or "gpt-4.1-mini"
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 또는 VISION_API_KEY가 필요합니다.")

    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VISION_API_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")

    extra_headers: Dict[str, str] = {}
    if "openai.com" in base_url:
        extra_headers["OpenAI-Beta"] = "assistants=v2"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers or None)

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        max_output_tokens=2048,
    )

    text = getattr(resp, "output_text", None)
    if text and isinstance(text, str) and text.strip():
        return text.strip()

    try:
        body = resp.model_dump()  # type: ignore[attr-defined]
    except Exception:
        try:
            body = resp.__dict__
        except Exception:
            body = {}

    analysis = _extract_text_from_response(body)
    if not analysis:
        raise RuntimeError("비전 API 응답에서 분석 결과를 찾을 수 없습니다.")
    return analysis


@cl.on_chat_start
async def on_chat_start():
    logger.info("Chat session started")
    cl.user_session.set("track_id", str(uuid4()))
    cl.user_session.set("current_text_msg", None)
    cl.user_session.set("current_image_name", None)
    cl.user_session.set("current_image_data_url", None)
    return True


@cl.on_message
async def on_message(message: cl.Message):
    try:
        # 이미지 처리
        files = message.elements
        if files:
            await _handle_image_upload(message, files)
            return

        # 텍스트 메시지 처리
        await _handle_text_message(message)

    except Exception as e:
        logger.error(f"Error in on_message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.ErrorMessage(
            content=f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        ).send()


async def _handle_image_upload(message: cl.Message, files):
    for file in files:
        try:
            file_name = file.name if hasattr(file, "name") else "알 수 없음"
            file_source = file.path if hasattr(file, "path") else file.url
            mime_type = getattr(file, "mime", None) or ""
            extension = Path(file_name).suffix.lower() if file_name else ""
            source_mime = (
                mimetypes.guess_type(file_source)[0]
                if isinstance(file_source, str)
                else ""
            )

            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
            is_image = mime_type.startswith("image/") or extension in image_extensions

            if not is_image and source_mime:
                is_image = source_mime.startswith("image/")

            image_bytes = None

            if is_image:
                image_bytes = await _read_file_bytes(file)
            else:
                try:
                    candidate_bytes = await _read_file_bytes(file)
                    with Image.open(io.BytesIO(candidate_bytes)) as pil_image:
                        detected_format = pil_image.format
                    if detected_format:
                        mime_type = Image.MIME.get(detected_format.upper(), "")
                        image_bytes = candidate_bytes
                        is_image = True
                except Exception:
                    image_bytes = None

            if is_image and image_bytes:
                if not mime_type:
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as pil_image:
                            detected_format = pil_image.format
                        if detected_format:
                            mime_type = Image.MIME.get(detected_format.upper(), "")
                    except Exception:
                        mime_type = ""

                if not mime_type:
                    mime_type = (
                        mimetypes.guess_type(file_name)[0]
                        or source_mime
                        or "image/jpeg"
                    )

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_data_url = f"data:{mime_type};base64,{image_b64}"

                user_message = (message.content or "").strip()
                prompt_sections = []
                if user_message:
                    prompt_sections.append(user_message)
                prompt_sections.append(_compose_guidance())
                prompt = "\n\n".join(prompt_sections)

                # 이미지 정보를 세션에 저장
                cl.user_session.set("current_image_name", file_name)
                cl.user_session.set("current_image_data_url", image_data_url)
                cl.user_session.set("is_text_input", True)

                try:
                    typing_msg = cl.Message(content="🔎 이미지 분석 중입니다… 잠시만 기다려 주세요.", author="assistant")
                    await typing_msg.send()
                    analysis_text = await _analyze_image_with_vision_api(
                        prompt, image_data_url
                    )
                    typing_msg.content = analysis_text
                    await typing_msg.update()
                except Exception as analysis_error:
                    logger.error(f"비전 API 이미지 분석 실패: {analysis_error}")
                    try:
                        typing_msg.content = f"⚠️ 이미지 분석 중 오류가 발생했습니다: {analysis_error}"
                        await typing_msg.update()
                    except Exception:
                        await cl.ErrorMessage(
                            content=f"이미지 분석 중 오류가 발생했습니다: {analysis_error}"
                        ).send()

                return

        except Exception as e:
            logger.error(f"이미지 처리 오류: {e}")
            logger.error(f"파일 정보: {file.__dict__}")
            await cl.ErrorMessage(
                content=f"이미지 처리 중 오류가 발생했습니다: {e}"
            ).send()


async def _handle_text_message(message: cl.Message):
    current_image_data_url = cl.user_session.get("current_image_data_url")

    if current_image_data_url:
        # 이미지 컨텍스트가 있는 경우 - 이미지 재분석
        try:
            prompt_sections = []
            if message.content:
                prompt_sections.append(message.content.strip())
            prompt_sections.append(_compose_guidance())
            analysis_prompt = "\n\n".join(filter(None, prompt_sections))

            cl.user_session.set("is_text_input", True)
            typing_msg = cl.Message(content="🔎 이미지 재분석 중입니다… 잠시만 기다려 주세요.", author="assistant")
            await typing_msg.send()
            analysis_text = await _analyze_image_with_vision_api(
                analysis_prompt, current_image_data_url
            )
            typing_msg.content = analysis_text
            await typing_msg.update()
        except Exception as analysis_error:
            logger.error(f"비전 API 이미지 분석 실패: {analysis_error}")
            try:
                typing_msg.content = f"⚠️ 이미지 분석 중 오류가 발생했습니다: {analysis_error}"
                await typing_msg.update()
            except Exception:
                await cl.ErrorMessage(
                    content=f"이미지 분석 중 오류가 발생했습니다: {analysis_error}"
                ).send()
    else:
        # 이미지가 없으면 안내 메시지 제공
        if_wrong_input_path = os.environ.get("PROMPT_IF_WRONG_INPUT_PATH", "prompts/if_wrong_input.md")
        tip= Path(if_wrong_input_path).read_text(encoding="utf-8").strip()

        await cl.Message(content=tip, author="assistant").send()
