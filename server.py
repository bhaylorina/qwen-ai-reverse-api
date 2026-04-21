"""FastAPI Server - OpenAI Compatible API for Qwen AI"""

import time
import random
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from qwen_ai import QwenAiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict]
    stream: bool = False
    temperature: Optional[float] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "qwen-ai"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class TokenHealthRequest(BaseModel):
    tokens: str


class TokenHealthResult(BaseModel):
    token: str
    status: str
    valid: bool
    error: Optional[str] = None


class TokenHealthResponse(BaseModel):
    total: int
    healthy: int
    unhealthy: int
    results: List[TokenHealthResult]


SUPPORTED_MODELS = ["qwen3.6-plus", "qwen3.5-plus", "qwen3.5-flash", "qwen3-max", "qwen3-coder", "qwen2.5-max"]

app = FastAPI(title="Qwen AI OpenAI API", version="0.3.0")


def select_random_token(token_string: str) -> str:
    tokens = [t.strip() for t in token_string.split(',') if t.strip()]
    if not tokens:
        raise ValueError("No valid tokens")
    return random.choice(tokens)


@app.get("/")
async def root():
    return {"service": "Qwen AI OpenAI API", "version": "0.3.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(data=[ModelInfo(id=m) for m in SUPPORTED_MODELS])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    if not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization")
    try:
        jwt_token = select_random_token(token)
        client = QwenAiClient(token=jwt_token)
        if request.stream:
            return StreamingResponse(
                client.chat_completions(request.model, request.messages, True, request.temperature, request.tools, request.tool_choice),
                media_type="text/event-stream")
        else:
            return JSONResponse(client.chat_completions(request.model, request.messages, False, request.temperature, request.tools, request.tool_choice))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tokens/health", response_model=TokenHealthResponse)
async def check_tokens_health(request: TokenHealthRequest):
    token_list = [t.strip() for t in request.tokens.split(',') if t.strip()]
    if not token_list:
        raise HTTPException(status_code=400, detail="No tokens provided")
    results = []
    healthy_count = 0
    for token in token_list:
        masked = token[:20] + "..." + token[-10:] if len(token) > 30 else token
        try:
            client = QwenAiClient(token=token)
            chat_id = client.create_chat('qwen3.5-plus', 'Health_Check')
            client.delete_chat(chat_id)
            results.append(TokenHealthResult(token=masked, status="healthy", valid=True))
            healthy_count += 1
        except Exception as e:
            results.append(TokenHealthResult(token=masked, status="unhealthy", valid=False, error=str(e)[:50]))
    return TokenHealthResponse(total=len(token_list), healthy=healthy_count, unhealthy=len(token_list) - healthy_count, results=results)


@app.get("/v1/tokens/health")
async def check_tokens_health_get(tokens: str):
    return await check_tokens_health(TokenHealthRequest(tokens=tokens))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
