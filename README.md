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

## 설치

1. 저장소를 클론합니다:
   ```bash
   git clone <repository-url>
   cd mcp-realtime-chainlit
   ```

2. 의존성 설치 (devcontainer에서는 자동 설치됨):
   ```bash
   pip install -e .
   ```

3. 환경 변수를 설정합니다 (`.env` 권장):
   ```
   # 기본(OpenAI) 설정 예시
   VISION_API_KEY=your_api_key
   VISION_MODEL=gpt-4.1-mini
   # 다른 공급자를 사용하려면 엔드포인트를 바꾸세요 (예: https://api.example.com/v1)
   # VISION_API_BASE_URL=https://api.openai.com/v1
   ```
   참고: 기존 OPENAI_* 변수도 하위 호환으로 인식됩니다 (OPENAI_API_KEY, OPENAI_VISION_MODEL).

4. 프롬프트 편집(선택):
   - `prompts/input.md`: 무엇을 어떤 관점으로 분석할지 정의
   - `prompts/output.md`: 사용자가 더 잘 찍을 수 있도록 안내할 출력 형식/가이드 정의
   - 환경 변수로 경로를 바꿀 수 있습니다:
     - `PROMPT_INPUT_PATH=custom/input.md`
     - `PROMPT_OUTPUT_PATH=custom/output.md`

## 사용법

애플리케이션을 시작합니다:

```bash
chainlit run chat.py
```

이것은 Chainlit 웹 인터페이스를 실행하며, 일반적으로 http://localhost:8000에서 접속할 수 있습니다.

### 사용 방법

- 음식 사진을 업로드하면 분석 결과가 텍스트로 표시됩니다.
- 업로드 후 추가 텍스트를 입력하면 같은 이미지 컨텍스트로 재분석합니다.

## 프로젝트 구조

- `chat.py`: 이미지 업로드 처리 및 비전 API 이미지 분석 호출
- `public/`: 정적 리소스

## 작동 원리

1. Chainlit 애플리케이션이 시작됩니다.
2. 사용자가 이미지를 업로드하면 Vision Responses API(OpenAI 호환)로 분석 요청을 보냅니다.
3. 분석 결과를 사용자에게 텍스트로 표시합니다.

## 개발

추가 개발 아이디어

- 결과 하이라이트(예: 구도/조명/배경 섹션별 강조)
- 간단한 점수화(0~5)와 체크리스트 제공
- 여러 장 이미지 비교 분석

## 라이선스

MIT 라이선스 (MIT)

## 감사의 말

- 이 프로젝트는 [Chainlit](https://github.com/Chainlit/chainlit) 프레임워크를 사용합니다
