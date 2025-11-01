# 음식 사진 분석 AI (Chainlit)
이 프로젝트는 사용자가 업로드한 음식 사진을 분석해 더 나은 사진을 찍을 수 있도록 피드백을 제공하는 최소 구현의 AI 웹 앱입니다. UI는 Chainlit를 사용하며, OpenAI 호환 Vision Responses API로 이미지를 분석하고 텍스트 결과를 보여줍니다.

## 기능
- **이미지 업로드 분석**: 음식 사진 업로드 시 즉시 분석 및 피드백 제공
- **텍스트 재질문 지원**: 같은 이미지 컨텍스트로 추가 질문/요청 시 재분석
- **간단한 구성**: 오디오/실시간/WebSocket/MCP 의존성 제거, 최소 동작에 집중

## 아키텍처
구성 요소:
1. **Chainlit 인터페이스 (`chat.py`)**: 이미지 업로드 처리, Vision Responses API 호출, 결과 표시

## 사전 요구사항
- Python 3.12+
- 비전 API 키 (기본값: OpenAI)

## 직접 실행을 위한 방법
1. 파이썬 가상환경 설정(uv 패키지 매니저 기반)
   ```bash
   uv sync
   ```

2. uv로 설정한 파이썬 가상환경 실행(WARNING: 이 명령어 실행 안하면 chainlit 명령을 찾을 수 없음)
   ```bash
   source ./venv/bin/active
   ```

## 사용법
chainlit 기반 애플리케이션을 시작합니다:

```bash
chainlit run chat.py
```

이것은 Chainlit 웹 인터페이스를 실행하며, 일반적으로 http://localhost:8000에서 접속할 수 있습니다.

### 사용 방법
- 음식 사진을 업로드하면 분석 결과가 텍스트로 표시됩니다.
- 업로드 후 추가 텍스트를 입력하면 같은 이미지 컨텍스트로 재분석합니다.

## 프로젝트 구조
- `chat.py`: 이미지 업로드 처리 및 비전 API 이미지 분석 호출

## 작동 원리
1. Chainlit 애플리케이션이 시작됩니다.
2. 사용자가 이미지를 업로드하면 Vision Responses API(OpenAI 호환)로 분석 요청을 보냅니다.
3. 분석 결과를 사용자에게 텍스트로 표시합니다.


## 라이선스
MIT 라이선스 (MIT)

## 감사의 말
- 이 프로젝트는 [Chainlit](https://github.com/Chainlit/chainlit) 프레임워크를 사용합니다
