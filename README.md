# Realtime Audio API + MCP Agent

이 프로젝트는 Model Context Protocol (MCP)과 OpenAI의 실시간 API를 사용하여 에어비앤비 정보를 제공하는 실시간 챗봇을 구현합니다. 이 챗봇은 Chainlit 웹 인터페이스를 통해 텍스트와 음성 상호작용을 모두 지원합니다.

## 기능

- **멀티모달 상호작용**: 텍스트 입력과 음성 대화를 모두 지원
- **실시간 오디오 처리**: 자연스러운 대화 흐름을 위한 오디오 스트리밍
- **Model Context Protocol (MCP) 통합**: 에어비앤비 데이터를 위한 외부 MCP 서비스 연결
- **다국어 지원**: 여러 언어 지원 (현재 영어와 한국어)
- **Azure OpenAI 통합**: 스트리밍 응답을 위한 Azure OpenAI의 실시간 API 사용

## 아키텍처

이 프로젝트는 세 가지 주요 구성 요소로 구성됩니다:

1. **MCP 서비스**: 에어비앤비 데이터를 제공하는 MCP 서버에 대한 연결을 관리
2. **실시간 클라이언트**: OpenAI의 실시간 API와 WebSocket 통신을 처리
3. **Chainlit 인터페이스**: 오디오 지원을 포함한 사용자 대면 채팅 인터페이스 제공

## 사전 요구사항

- Python 3.11+
- Node.js와 npm (MCP 서버용)
- Python용 uv 패키지 매니저
- 실시간 기능을 가진 Azure OpenAI API 액세스

## 설치

1. 저장소를 클론합니다:
   ```bash
   git clone <repository-url>
   cd mcp-realtime-chainlit
   ```

2. uv를 사용하여 Python 의존성을 설치합니다 (현대적인 Python 패키지 매니저):
   ```bash
   uv sync
   ```
   
   참고: uv가 설치되어 있지 않다면, 먼저 설치하세요:
   ```bash
   curl -sSf https://install.ultraviolet.rs | sh
   ```

3. Azure OpenAI 자격 증명으로 `.env` 파일을 생성합니다:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   ```

## 사용법

애플리케이션을 시작합니다:

```bash
chainlit run chat.py
```

이것은 Chainlit 웹 인터페이스를 실행하며, 일반적으로 http://localhost:8000에서 접속할 수 있습니다.

### 챗봇과 상호작용하기

- **텍스트 모드**: 에어비앤비 목록, 가격, 위치 등에 대한 질문을 입력하세요.
- **음성 모드**: 마이크 버튼을 클릭하여 말하기를 시작하면, 시스템이 음성 입력을 처리합니다.

## 프로젝트 구조

- `mcp_service.py`: MCP 서버와 통신하기 위한 MCP 서비스 클라이언트 구현
- `realtime.py`: OpenAI 실시간 API에 대한 WebSocket 연결 관리
- `chat.py`: 텍스트와 오디오를 위한 핸들러가 포함된 Chainlit 인터페이스 구현
- `chainlit_config.py`: 다국어 지원을 위한 설정
- `locales/`: 다양한 언어의 번역 파일

## 작동 원리

1. Chainlit 애플리케이션이 시작되고 Azure OpenAI의 실시간 API에 연결을 설정합니다
2. MCP 서비스가 초기화되고 에어비앤비 MCP 서버에 연결됩니다
3. 사용자가 메시지(텍스트 또는 오디오)를 보낼 때:
   - 텍스트인 경우: 메시지가 모델에 직접 전송됩니다
   - 오디오인 경우: 오디오가 실시간 전사를 위해 모델로 스트리밍됩니다
4. 모델은 MCP 도구를 통해 에어비앤비 데이터에 액세스하여 입력을 처리합니다
5. 응답이 사용자에게 스트리밍됩니다 (텍스트 및/또는 오디오)

## 개발

### 새로운 MCP 도구 추가하기

새로운 MCP 서버를 추가하려면:

1. `MCPService.initialize()` 메서드에 서버 설정을 추가합니다
2. MCP 서버에 필요한 npm 패키지를 설치합니다

## 라이선스

MIT 라이선스 (MIT)

## 감사의 말

- 이 프로젝트는 채팅 인터페이스를 위해 [Chainlit](https://github.com/Chainlit/chainlit) 프레임워크를 사용합니다
- 실시간 스트리밍 구현은 [openai-realtime-console](https://github.com/openai/openai-realtime-console)에서 파생되었습니다
- 에어비앤비 데이터는 [@openbnb/mcp-server-airbnb](https://www.npmjs.com/package/@openbnb/mcp-server-airbnb)를 통해 제공됩니다
