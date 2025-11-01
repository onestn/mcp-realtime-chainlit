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
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI

# chainlit
import chainlit as cl
from chainlit.logger import logger


load_dotenv(override=True)

# 음식 카테고리 정의
FOOD_CATEGORIES = {
    "hamburger": ["햄버거", "버거", "burger", "hamburger"],
    "meat": ["고기", "삼겹살", "스테이크", "meat", "steak", "pork", "beef", "갈비", "구이"],
    "noodles": ["면", "라면", "파스타", "라멘", "noodle", "ramen", "pasta", "spaghetti", "우동", "국수"]
}


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
    logger.info("_read_file_bytes 시작")
    
    # content 속성 확인
    if hasattr(file, "content") and file.content:
        logger.info("file.content 발견")
        if isinstance(file.content, (bytes, bytearray)):
            logger.info(f"bytes/bytearray 형식: {len(file.content)} bytes")
            return bytes(file.content)
        if hasattr(file.content, "read"):
            logger.info("file.content.read() 사용")
            data = file.content.read()
            logger.info(f"읽은 데이터: {len(data)} bytes")
            return data

    # path 속성 확인
    file_path = getattr(file, "path", None)
    if file_path:
        logger.info(f"file.path 발견: {file_path}")
        if os.path.exists(file_path):
            logger.info(f"파일 존재 확인, 읽기 시작")
            data = await asyncio.to_thread(Path(file_path).read_bytes)
            logger.info(f"파일 읽기 완료: {len(data)} bytes")
            return data
        else:
            logger.warning(f"파일이 존재하지 않음: {file_path}")

    # url 속성 확인
    file_url = getattr(file, "url", None)
    if file_url:
        logger.info(f"file.url 발견: {file_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                logger.info(f"URL에서 다운로드 완료: {len(data)} bytes")
                return data

    logger.error("이미지 데이터를 읽을 수 있는 방법이 없음")
    logger.error(f"파일 속성: {dir(file)}")
    raise ValueError("이미지 데이터를 읽을 수 없습니다.")


def _detect_food_category(analysis_text: str) -> Optional[str]:
    """분석 텍스트에서 음식 카테고리를 감지합니다.
    
    Returns:
        카테고리 키 ("hamburger", "meat", "noodles") 또는 None
    """
    text_lower = analysis_text.lower()
    
    for category, keywords in FOOD_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                logger.info(f"음식 카테고리 감지: {category} (키워드: {keyword})")
                return category
    
    logger.info("음식 카테고리를 감지하지 못했습니다.")
    return None


async def _load_reference_images(category: str) -> Tuple[List[str], List[str]]:
    """해당 카테고리의 참조 이미지들을 로드합니다.
    
    Args:
        category: "hamburger", "meat", "noodles" 중 하나
        
    Returns:
        (good_image_urls, bad_image_urls) 튜플
    """
    good_images: List[str] = []
    bad_images: List[str] = []
    
    base_path = Path("public/reference_images") / category
    
    # good 폴더의 이미지 로드
    good_dir = base_path / "good"
    if good_dir.exists():
        for img_path in good_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
                try:
                    img_bytes = await asyncio.to_thread(img_path.read_bytes)
                    # MIME 타입 추론
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            mime_type = Image.MIME.get(pil_img.format, "image/jpeg")
                    except Exception:
                        mime_type = "image/jpeg"
                    
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{img_b64}"
                    good_images.append(data_url)
                    logger.info(f"참조 이미지 로드 (good): {img_path.name}")
                except Exception as e:
                    logger.error(f"참조 이미지 로드 실패 (good/{img_path.name}): {e}")
    
    # bad 폴더의 이미지 로드
    bad_dir = base_path / "bad"
    if bad_dir.exists():
        for img_path in bad_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
                try:
                    img_bytes = await asyncio.to_thread(img_path.read_bytes)
                    # MIME 타입 추론
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            mime_type = Image.MIME.get(pil_img.format, "image/jpeg")
                    except Exception:
                        mime_type = "image/jpeg"
                    
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{img_b64}"
                    bad_images.append(data_url)
                    logger.info(f"참조 이미지 로드 (bad): {img_path.name}")
                except Exception as e:
                    logger.error(f"참조 이미지 로드 실패 (bad/{img_path.name}): {e}")
    
    logger.info(f"카테고리 '{category}': good {len(good_images)}장, bad {len(bad_images)}장 로드 완료")
    return good_images, bad_images


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


async def _analyze_with_reference_images(
    user_image_urls: List[str],
    category: str,
    base_prompt: str
) -> str:
    """사용자 이미지와 참조 이미지를 함께 비교 분석합니다.
    
    Args:
        user_image_urls: 사용자가 업로드한 이미지 data URL 리스트
        category: 음식 카테고리 ("hamburger", "meat", "noodles")
        base_prompt: 기본 프롬프트
        
    Returns:
        비교 분석 결과 텍스트
    """
    # 참조 이미지 로드
    good_refs, bad_refs = await _load_reference_images(category)
    
    if not good_refs and not bad_refs:
        logger.warning(f"카테고리 '{category}'의 참조 이미지가 없습니다. 일반 분석만 수행합니다.")
        if len(user_image_urls) == 1:
            return await _analyze_image_with_vision_api(base_prompt, user_image_urls[0])
        else:
            return await _analyze_images_with_vision_api(base_prompt, user_image_urls)
    
    # 비교 분석 프롬프트 구성
    category_names = {
        "hamburger": "햄버거",
        "meat": "고기류",
        "noodles": "면류"
    }
    category_name = category_names.get(category, category)
    
    comparison_prompt = f"""당신은 음식 사진 전문가입니다. 사용자가 업로드한 {category_name} 사진을 분석하고, 제공된 참조 이미지들과 비교하여 구체적인 피드백을 제공해주세요.

**분석 순서:**
1. 먼저 사용자의 사진을 분석합니다.
2. "잘 찍힌 예시" 이미지들과 비교하여 어떤 점이 좋고 어떤 점이 부족한지 설명합니다.
3. "못 찍힌 예시" 이미지들과 비교하여 유사한 실수를 피하기 위한 조언을 제공합니다.
4. 구체적이고 실천 가능한 개선 방법을 제시합니다.

**평가 기준:**
- 구도 및 프레이밍
- 조명 (자연광/인공광)
- 초점 및 선명도
- 배경 정리 및 분위기
- 색감 및 대비

**사용자 요청:**
{base_prompt}

**참조 이미지 구성:**
- 잘 찍힌 예시: {len(good_refs)}장
- 못 찍힌 예시: {len(bad_refs)}장

아래 순서대로 이미지가 제공됩니다:
1. 사용자 사진 ({len(user_image_urls)}장)
2. 잘 찍힌 예시 ({len(good_refs)}장)
3. 못 찍힌 예시 ({len(bad_refs)}장)
"""
    
    # API 키 및 모델 설정
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

    # content 파트 구성: prompt + 사용자 이미지 + 참조 이미지(good) + 참조 이미지(bad)
    content_parts: List[Dict[str, Any]] = [{"type": "input_text", "text": comparison_prompt}]
    
    # 사용자 이미지 추가
    for url in user_image_urls:
        content_parts.append({"type": "input_image", "image_url": url})
    
    # 잘 찍힌 참조 이미지 추가
    for url in good_refs:
        content_parts.append({"type": "input_image", "image_url": url})
    
    # 못 찍힌 참조 이미지 추가
    for url in bad_refs:
        content_parts.append({"type": "input_image", "image_url": url})
    
    logger.info(f"비교 분석 요청: 사용자 {len(user_image_urls)}장 + 참조(good) {len(good_refs)}장 + 참조(bad) {len(bad_refs)}장")

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content_parts,
            }
        ],
        max_output_tokens=3072,  # 비교 분석은 더 긴 응답이 필요할 수 있음
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


async def _analyze_images_with_vision_api(prompt: str, image_data_urls: List[str]) -> str:
    """여러 장의 이미지를 한 번에 분석합니다.

    - prompt: 사용자 프롬프트 + 가이드라인
    - image_data_urls: data:<mime>;base64,<payload> 형태의 URL 리스트

    하나의 요청으로 input_text 다음에 여러 개의 input_image 파트를 붙여 전송합니다.
    """
    if not image_data_urls:
        raise ValueError("분석할 이미지가 없습니다.")

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

    # content 파트 구성: 첫 파트는 input_text, 이어서 모든 input_image
    content_parts: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for url in image_data_urls:
        content_parts.append({"type": "input_image", "image_url": url})

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content_parts,
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
    cl.user_session.set("current_image_data_url", None)  # 하위 호환(단일 이미지)
    cl.user_session.set("current_image_names", [])
    cl.user_session.set("current_image_data_urls", [])
    return True


@cl.on_message
async def on_message(message: cl.Message):
    try:
        # 이미지 처리 - elements 속성과 첨부 파일 모두 확인
        files = []
        
        # message.elements 확인 (Chainlit 구버전)
        if hasattr(message, 'elements') and message.elements:
            files = message.elements
            logger.info(f"message.elements에서 {len(files)}개 파일 발견")
        
        # attachments 확인 (Chainlit 신버전)
        if not files and hasattr(message, 'attachments') and message.attachments:
            files = message.attachments
            logger.info(f"message.attachments에서 {len(files)}개 파일 발견")
        
        # 파일이 있으면 이미지 처리
        if files:
            logger.info(f"총 {len(files)}개 파일 처리 시작")
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
    image_entries: List[Dict[str, Any]] = []  # {name, url}
    
    logger.info(f"_handle_image_upload 호출됨: {len(files)}개 파일")

    # 1) 업로드된 파일들 중 이미지만 추출하여 data URL 리스트로 변환
    for idx, file in enumerate(files):
        try:
            logger.info(f"파일 {idx+1} 처리 중...")
            
            # 파일 속성 확인
            file_name = getattr(file, "name", None) or getattr(file, "filename", None) or "알 수 없음"
            file_path = getattr(file, "path", None)
            file_url = getattr(file, "url", None)
            mime_type = getattr(file, "mime", None) or getattr(file, "type", None) or ""
            
            logger.info(f"파일명: {file_name}, MIME: {mime_type}, 경로: {file_path}, URL: {file_url}")
            
            file_source = file_path if file_path else file_url
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
                # MIME이 비어있어도 실제로 이미지일 수 있으니 시도해 본다.
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
                logger.info(f"이미지로 확인됨: {file_name} ({len(image_bytes)} bytes)")
                
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

                image_entries.append({"name": file_name, "url": image_data_url})
                logger.info(f"이미지 추가 완료: {file_name}")
            else:
                logger.warning(f"이미지가 아니거나 읽기 실패: {file_name}")

        except Exception as e:
            logger.error(f"이미지 처리 오류: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            try:
                logger.error(f"파일 정보: {getattr(file, '__dict__', {})}")
            except Exception:
                pass

    # 2) 최소 1장 이상인지 확인
    logger.info(f"총 {len(image_entries)}개 이미지 추출 완료")
    
    if not image_entries:
        await cl.ErrorMessage(content="이미지 파일을 찾지 못했습니다. 이미지 파일을 업로드해 주세요.").send()
        return

    # 3) 프롬프트 구성
    user_message = (message.content or "").strip()
    prompt_sections = []
    if user_message:
        prompt_sections.append(user_message)
    prompt_sections.append(_compose_guidance())
    prompt = "\n\n".join(prompt_sections)

    # 4) 세션에 저장 (리스트)
    names = [e["name"] for e in image_entries]
    urls = [e["url"] for e in image_entries]
    cl.user_session.set("current_image_names", names)
    cl.user_session.set("current_image_data_urls", urls)

    # 하위 호환: 단일 키에도 첫 장을 기록
    cl.user_session.set("current_image_name", names[0])
    cl.user_session.set("current_image_data_url", urls[0])
    cl.user_session.set("is_text_input", True)

    # 5) 1차 분석 요청
    try:
        count = len(urls)
        prefix = f"🔎 이미지 {count}장 분석 중입니다…" if count > 1 else "🔎 이미지 분석 중입니다…"
        typing_msg = cl.Message(content=f"{prefix} 잠시만 기다려 주세요.", author="assistant")
        await typing_msg.send()

        # 1차 분석 (카테고리 감지를 위해)
        if len(urls) == 1:
            initial_analysis = await _analyze_image_with_vision_api(prompt, urls[0])
        else:
            initial_analysis = await _analyze_images_with_vision_api(prompt, urls)

        typing_msg.content = initial_analysis
        await typing_msg.update()
        
        # 6) 음식 카테고리 감지 및 참조 이미지 비교 분석
        detected_category = _detect_food_category(initial_analysis)
        
        if detected_category:
            logger.info(f"음식 카테고리 '{detected_category}' 감지됨. 참조 이미지 비교 분석을 시작합니다.")
            
            # 비교 분석 진행 메시지
            comparison_msg = cl.Message(
                content=f"📸 참조 이미지와 비교 분석 중입니다… 잠시만 기다려 주세요.",
                author="assistant"
            )
            await comparison_msg.send()
            
            try:
                # 참조 이미지와 함께 비교 분석
                comparison_result = await _analyze_with_reference_images(
                    user_image_urls=urls,
                    category=detected_category,
                    base_prompt=prompt
                )
                
                comparison_msg.content = f"## 📊 참조 이미지 비교 분석 결과\n\n{comparison_result}"
                await comparison_msg.update()
                
            except Exception as comp_error:
                logger.error(f"참조 이미지 비교 분석 실패: {comp_error}")
                comparison_msg.content = f"⚠️ 참조 이미지 비교 분석 중 오류가 발생했습니다: {comp_error}"
                await comparison_msg.update()
        else:
            logger.info("특정 음식 카테고리를 감지하지 못했습니다. 일반 분석 결과만 제공합니다.")
            
    except Exception as analysis_error:
        logger.error(f"비전 API 이미지 분석 실패: {analysis_error}")
        try:
            typing_msg.content = f"⚠️ 이미지 분석 중 오류가 발생했습니다: {analysis_error}"
            await typing_msg.update()
        except Exception:
            await cl.ErrorMessage(
                content=f"이미지 분석 중 오류가 발생했습니다: {analysis_error}"
            ).send()


async def _handle_text_message(message: cl.Message):
    urls: List[str] = cl.user_session.get("current_image_data_urls") or []
    single_url = cl.user_session.get("current_image_data_url")

    # 단일 키만 존재하고 리스트가 비어있다면 보강
    if not urls and single_url:
        urls = [single_url]

    if urls:
        # 이미지 컨텍스트가 있는 경우 - 이미지 재분석
        try:
            prompt_sections = []
            if message.content:
                prompt_sections.append(message.content.strip())
            prompt_sections.append(_compose_guidance())
            analysis_prompt = "\n\n".join(filter(None, prompt_sections))

            cl.user_session.set("is_text_input", True)
            count = len(urls)
            prefix = f"🔎 이미지 {count}장 재분석 중입니다…" if count > 1 else "🔎 이미지 재분석 중입니다…"
            typing_msg = cl.Message(content=f"{prefix} 잠시만 기다려 주세요.", author="assistant")
            await typing_msg.send()

            if len(urls) == 1:
                analysis_text = await _analyze_image_with_vision_api(
                    analysis_prompt, urls[0]
                )
            else:
                analysis_text = await _analyze_images_with_vision_api(
                    analysis_prompt, urls
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
        tip = Path(if_wrong_input_path).read_text(encoding="utf-8").strip()
        await cl.Message(content=tip, author="assistant").send()
