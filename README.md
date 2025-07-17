# Chainlit Realtime MCP

A Chainlit application that combines OpenAI's Realtime API with Model Context Protocol (MCP) integration for building voice-based AI assistants.

## Features

- Voice-based conversations via OpenAI's Realtime API
- Tool integration through Model Context Protocol (MCP)
- Multi-language support (English, Korean)
- Streaming transcript and audio response

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or use uv:
   ```bash
   uv pip install -e .
   ```

## Configuration

1. Copy `.env.sample` to `.env` and fill in the required environment variables:
   ```
   AZURE_OPENAI_ENDPOINT=wss://<RESOURCE>.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT=gpt-4o-realtime-preview
   AZURE_OPENAI_API_KEY=<leave empty if you want to use Entra ID Authn, RECOMMENDED>
   ```

2. Language settings are configured in `chainlit_config.py`. The application currently supports:
   - English (en) - default
   - Korean (ko)

## Running the Application

Start the Chainlit application with:

```bash
chainlit run chat.py
```

## Project Structure

- `chat.py` - Main Chainlit application
- `realtime.py` - OpenAI Realtime API client
- `mcp_service.py` - Model Context Protocol service
- `assistant_service.py` - Assistant service for agent-based conversations
- `chainlit_config.py` - Chainlit configuration including translations
- `locales/` - Translation files for internationalization
  - `en/` - English translations
  - `ko/` - Korean translations

## Adding New Languages

To add a new language:

1. Create a new directory in `locales/` with the language code (e.g., `ja` for Japanese)
2. Copy the `translation.json` from an existing language and translate all values
3. Update `supported_languages` in `chainlit_config.py` to include the new language code

## MCP Integration

The application includes MCP integration for tool calls. The default configuration includes:

- Airbnb MCP server for Airbnb-related functionalities

Additional MCP servers can be configured in `.vscode/settings.json` and `mcp_service.py`.
