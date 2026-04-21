"""Qwen AI Adapter for chat.qwen.ai"""

import json
import uuid
import time
import logging
import requests
from typing import Dict, Optional, Tuple, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class QwenAiAdapter:
    QWEN_AI_BASE = 'https://chat.qwen.ai'
    DEFAULT_TIMEOUT = 120
    CONNECT_TIMEOUT = 30
    
    DEFAULT_HEADERS = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'source': 'web',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
        'Origin': 'https://chat.qwen.ai',
    }
    
    MODEL_ALIASES = {
        'qwen': 'qwen3-max', 'qwen3': 'qwen3-max', 'qwen3.5': 'qwen3.5-plus',
        'qwen3-coder': 'qwen3-coder-plus', 'qwen2.5': 'qwen2.5-max',
    }
    
    VALID_MODELS = {
        "qwen3.6-plus", "qwen3.5-plus", "qwen3.5-flash", "qwen3-max",
        "qwen3-coder", "qwen3-coder-plus", "qwen2.5-max",
    }

    def __init__(self, token: str, cookies: Optional[str] = None):
        if not token:
            raise ValueError("Token required")
        self.token = token
        self.cookies = cookies
        self._force_thinking = None
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def get_headers(self, chat_id: Optional[str] = None) -> Dict[str, str]:
        headers = {**self.DEFAULT_HEADERS, 'Authorization': f'Bearer {self.token}', 'X-Request-Id': self._uuid()}
        if chat_id:
            headers['Referer'] = f'{self.QWEN_AI_BASE}/c/{chat_id}'
        if self.cookies:
            headers['Cookie'] = self.cookies
        return headers

    def map_model(self, openai_model: str) -> str:
        model = openai_model.lower().strip()
        if model in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model]
        if model in self.VALID_MODELS:
            return model
        return "qwen3.5-plus"

    def create_chat(self, model_id: str, title: str = 'New Chat') -> str:
        url = f'{self.QWEN_AI_BASE}/api/v2/chats/new'
        payload = {'title': title, 'models': [model_id], 'chat_mode': 'normal', 'chat_type': 't2t', 'timestamp': int(time.time() * 1000)}
        response = self.session.post(url, json=payload, headers=self.get_headers(), timeout=self.CONNECT_TIMEOUT)
        response.raise_for_status()
        return response.json()['data']['id']

    def delete_chat(self, chat_id: str) -> bool:
        if not chat_id:
            return False
        url = f'{self.QWEN_AI_BASE}/api/v2/chats/{chat_id}'
        response = self.session.delete(url, headers=self.get_headers(), timeout=self.CONNECT_TIMEOUT)
        return response.json().get('success', False)

    def delete_all_chats(self) -> bool:
        url = f'{self.QWEN_AI_BASE}/api/v2/chats/'
        response = self.session.delete(url, headers=self.get_headers(), timeout=self.CONNECT_TIMEOUT)
        return response.json().get('success', False)

    def chat_completion(self, model: str, messages: list, stream: bool = True, temperature: Optional[float] = None, enable_thinking: Optional[bool] = None, thinking_budget: Optional[int] = None) -> Tuple[requests.Response, str, Optional[str]]:
        model_id = self.map_model(model)
        chat_id = self.create_chat(model_id, 'API_Chat')
        user_content = self._build_message_content(messages)
        feature_config = {'thinking_enabled': enable_thinking or False, 'output_schema': 'phase', 'auto_search': False}
        if thinking_budget:
            feature_config['thinking_budget'] = thinking_budget
        payload = {
            'stream': True, 'version': '2.1', 'incremental_output': True,
            'chat_id': chat_id, 'model': model_id,
            'messages': [{
                'fid': self._uuid(), 'role': 'user', 'content': user_content,
                'models': [model_id], 'chat_type': 't2t', 'feature_config': feature_config,
            }],
        }
        url = f'{self.QWEN_AI_BASE}/api/v2/chat/completions?chat_id={chat_id}'
        response = self.session.post(url, json=payload, headers={**self.get_headers(chat_id), 'x-accel-buffering': 'no'}, stream=True, timeout=self.DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response, chat_id, None

    def _build_message_content(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                parts.insert(0, content)
            elif role == 'user':
                parts.append(f"User: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
        return '\n\n'.join(parts)
