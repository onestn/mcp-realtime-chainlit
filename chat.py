import asyncio
import base64
import io
import mimetypes
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import chainlit as cl
from chainlit.logger import logger
from dotenv import load_dotenv
from PIL import Image

from realtime import RealtimeClient

load_dotenv(override=True)


IMAGE_ANALYSIS_GUIDANCE = """
이 음식 사진을 전문적으로 분석해주세요. 다음 관점에서 구체적인 피드백을 제공해주세요:

1. 구도와 각도: 음식의 매력이 잘 드러나는지, 요리의 특징이 잘 보이는지
2. 조명: 그림자가 적절한지, 음식의 색감과 질감이 살아있는지
3. 배경과 구성: 주변 요소들이 음식을 돋보이게 하는지
4. 거리감과 포커스: 음식의 볼륨감과 디테일이 잘 표현되었는지
5. 개선 제안: 더 나은 사진을 위한 구체적이고 실행 가능한 조언

답변은 건설적이고 친절하게 작성해주세요.
"""


async def _read_file_bytes(file) -> bytes:
    """Best effort helper to extract raw bytes from a Chainlit file element."""

    if hasattr(file, "content") and file.content:
        # Chainlit already gives us the bytes in memory.
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


def _extract_text_from_openai_response(response: Dict[str, Any]) -> str:
    """Extract concatenated text segments from OpenAI Responses payload."""

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
                # Nested message content (OpenAI sometimes wraps messages)
                nested_content = part.get("content", [])
                for nested in nested_content:
                    if nested.get("type") in {"output_text", "text"}:
                        collected.append(nested.get("text", ""))

    text = "\n".join(filter(None, (segment.strip() for segment in collected)))
    return text.strip()


async def _analyze_image_with_openai(prompt: str, image_data_url: str) -> str:
    """Send image + prompt to OpenAI Responses API and return assistant text."""

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

    url = "https://api.openai.com/v1/responses"

    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        "max_output_tokens": 1024,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "assistants=v2",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 400:
                logger.error(f"OpenAI 이미지 분석 실패: {body}")
                resp.raise_for_status()

    analysis = _extract_text_from_openai_response(body)
    if not analysis:
        raise RuntimeError("OpenAI 응답에서 분석 결과를 찾을 수 없습니다.")
    return analysis


async def setup_openai_realtime():
    """음식 사진 분석을 위한 OpenAI Realtime Client 설정"""
    system_prompt = """
당신은 음식 사진 전문 분석가입니다. 

다음과 같은 관점에서 음식 사진을 분석하고 피드백을 제공하세요:
- 구도와 각도: 음식의 매력이 잘 드러나는지 평가
- 조명: 그림자와 색감이 적절한지 분석
- 배경과 구성: 주변 요소들이 음식을 돋보이게 하는지 검토
- 거리감과 포커스: 음식의 볼륨감과 디테일 표현 평가

피드백은 건설적이고 구체적으로 제공하며, 개선점 제안 시 실행 가능한 방법을 함께 제시하세요.

답변 형식:
1. 현재 사진의 장점
2. 개선 가능한 부분
3. 더 나은 사진을 위한 실천적 조언
    """
    openai_realtime = RealtimeClient(system_prompt=system_prompt, max_tokens=4096)

    cl.user_session.set("track_id", str(uuid4()))
    cl.user_session.set("is_text_input", True)

    async def handle_conversation_updated(event):
        """대화 업데이트 이벤트 처리 - 텍스트 응답 스트리밍"""
        item = event.get("item")
        delta = event.get("delta")

        try:
            if delta and "text" in delta:
                text = delta["text"]
                if item and item.get("role") == "assistant":
                    text_msg = cl.user_session.get("current_text_msg")
                    if not text_msg:
                        logger.info("Text response started")
                        text_msg = cl.Message(content="", author="assistant")
                        cl.user_session.set("current_text_msg", text_msg)
                        await text_msg.send()

                    text_msg.content += text
                    await text_msg.update()
        except Exception as e:
            logger.error(f"Error in handle_conversation_updated: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e

    async def handle_item_completed(item):
        """대화 아이템 완료 이벤트 처리"""
        try:
            if item["item"]["type"] == "message":
                content = item["item"]["content"][0]
                if content["type"] == "text":
                    logger.info("Text response completed")
                    text_msg = cl.user_session.get("current_text_msg")
                    cl.user_session.set("current_text_msg", None)

                    if text_msg and text_msg.content:
                        text_msg.content = content.get("text", text_msg.content)
                        await text_msg.update()
                    else:
                        final_msg = cl.Message(
                            content=content.get("text", ""), author="assistant"
                        )
                        await final_msg.send()
        except Exception as e:
            logger.error(f"Error in handle_item_completed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e

    async def handle_conversation_interrupt(event):
        """대화 중단 이벤트 처리"""
        cl.user_session.set("track_id", str(uuid4()))

    async def handle_response_done(event):
        """응답 완료 이벤트 처리"""
        cl.user_session.set("is_text_input", False)
        cl.user_session.set("current_text_msg", None)

    async def handle_error(event):
        logger.error(f"Realtime connection error: {event}")
        await cl.ErrorMessage(content=f"연결 오류가 발생했습니다: {event}").send()

    openai_realtime.on("conversation.updated", handle_conversation_updated)
    openai_realtime.on("conversation.item.completed", handle_item_completed)
    openai_realtime.on("conversation.interrupted", handle_conversation_interrupt)
    openai_realtime.on("server.response.done", handle_response_done)
    openai_realtime.on("error", handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)


@cl.on_chat_start
async def start():
    """채팅 세션 시작"""
    logger.info("Chat session started")
    try:
        await setup_openai_realtime()

        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        if not openai_realtime:
            logger.error("Failed to get openai_realtime from session")
            await cl.ErrorMessage(
                content="OpenAI 클라이언트 초기화에 실패했습니다"
            ).send()
            return False

        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        return True
    except Exception as e:
        logger.error(f"Error in start function: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.ErrorMessage(content=f"OpenAI 연결에 실패했습니다: {e}").send()
        return False


@cl.on_message
async def on_message(message: cl.Message):
    """메시지 처리 - 이미지 또는 텍스트"""
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")

        if not openai_realtime:
            logger.error("OpenAI realtime client not found in session")
            await cl.ErrorMessage(
                content="OpenAI 클라이언트가 초기화되지 않았습니다"
            ).send()
            return

        if not openai_realtime.is_connected():
            logger.error("OpenAI realtime client not connected")
            await cl.ErrorMessage(content="OpenAI에 연결되지 않았습니다").send()
            return

        # 이미지 처리
        files = message.elements
        if files:
            await _handle_image_upload(message, files)
            return

        # 텍스트 메시지 처리
        await _handle_text_message(message, openai_realtime)

    except Exception as e:
        logger.error(f"Error in on_message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await cl.ErrorMessage(
            content=f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        ).send()


async def _handle_image_upload(message: cl.Message, files):
    """이미지 업로드 처리"""
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
                prompt_sections.append(IMAGE_ANALYSIS_GUIDANCE)
                prompt = "\n\n".join(prompt_sections)

                # 이미지 정보를 세션에 저장
                cl.user_session.set("current_image_name", file_name)
                cl.user_session.set("current_image_data_url", image_data_url)
                cl.user_session.set("is_text_input", True)

                try:
                    analysis_text = await _analyze_image_with_openai(
                        prompt, image_data_url
                    )
                    await cl.Message(content=analysis_text, author="assistant").send()
                except Exception as analysis_error:
                    logger.error(f"OpenAI 이미지 분석 실패: {analysis_error}")
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


async def _handle_text_message(message: cl.Message, openai_realtime: RealtimeClient):
    """텍스트 메시지 처리"""
    current_image_data_url = cl.user_session.get("current_image_data_url")

    if current_image_data_url:
        # 이미지 컨텍스트가 있는 경우 - 이미지 재분석
        try:
            prompt_sections = []
            if message.content:
                prompt_sections.append(message.content.strip())
            prompt_sections.append(IMAGE_ANALYSIS_GUIDANCE)
            analysis_prompt = "\n\n".join(filter(None, prompt_sections))

            cl.user_session.set("is_text_input", True)
            analysis_text = await _analyze_image_with_openai(
                analysis_prompt, current_image_data_url
            )
            await cl.Message(content=analysis_text, author="assistant").send()
        except Exception as analysis_error:
            logger.error(f"OpenAI 이미지 분석 실패: {analysis_error}")
            await cl.ErrorMessage(
                content=f"이미지 분석 중 오류가 발생했습니다: {analysis_error}"
            ).send()
    else:
        # 일반 텍스트 대화
        cl.user_session.set("is_text_input", True)
        await openai_realtime.update_session(modalities=["text"])
        message_content = [{"type": "input_text", "text": message.content}]
        await openai_realtime.send_user_message_content(message_content)


# 오디오 관련 핸들러는 음식 사진 분석 서비스에 불필요하므로 제거됨
