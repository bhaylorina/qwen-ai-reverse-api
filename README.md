# Qwen AI OpenAI Compatible API

OpenAI-compatible API wrapper for Qwen AI (chat.qwen.ai).

## Features

- ✅ OpenAI SDK Compatible
- ✅ Streaming Support
- ✅ Multi-turn Conversation
- ✅ Tool/Function Calling
- ✅ Thinking/Reasoning Mode
- ✅ Multiple Token Load Balancing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python server.py
# or
python start_server.py --host 0.0.0.0 --port 8000
```

## Get JWT Token

1. Visit https://chat.qwen.ai and login
2. Press F12 → Application → Local Storage → chat.qwen.ai
3. Copy the `token` value

## API Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_JWT_TOKEN"
)

response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Supported Models

- qwen3.6-plus
- qwen3.5-plus
- qwen3.5-flash
- qwen3-max
- qwen3-coder
- qwen2.5-max

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/v1/tokens/health` | POST | Check token health |
| `/health` | GET | Health check |

## License

MIT
