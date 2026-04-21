"""Qwen AI Client - OpenAI Compatible Interface"""

import copy
import logging
from typing import List, Dict, Optional, Any, Generator, Union
from .adapter import QwenAiAdapter
from .stream_handler import QwenAiStreamHandler
from .tool_parser import ToolParser

logger = logging.getLogger(__name__)


class QwenAiClient:
    def __init__(self, token: str, cookies: Optional[str] = None):
        if not token:
            raise ValueError("Token is required")
        self.token = token
        self.cookies = cookies
        self.adapter = QwenAiAdapter(token, cookies)
        self._active_chats: Dict[str, Dict] = {}

    def chat_completions(self, model: str, messages: List[Dict], stream: bool = False,
                         temperature: Optional[float] = None, tools: Optional[List[Dict]] = None,
                         tool_choice: Optional[Any] = None, enable_thinking: Optional[bool] = None,
                         thinking_budget: Optional[int] = None, **kwargs) -> Union[Dict, Generator[str, None, None]]:
        if not model:
            raise ValueError("Model is required")
        if not messages:
            raise ValueError("Messages required")
        processed_messages = self._prepare_messages(messages, tools)
        response, chat_id, parent_id = self.adapter.chat_completion(
            model=model, messages=processed_messages, stream=stream, temperature=temperature,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget)
        handler = QwenAiStreamHandler(model, lambda cid: self._cleanup_chat(cid))
        handler.set_chat_id(chat_id)
        self._active_chats[chat_id] = {"model": model, "messages": messages, "chat_id": chat_id}
        if stream:
            return handler.handle_stream(response)
        else:
            result = handler.handle_non_stream(response)
            result["chat_id"] = chat_id
            return result

    def _prepare_messages(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> List[Dict]:
        processed = copy.deepcopy(messages)
        for i, msg in enumerate(processed):
            role = msg.get("role", "")
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls_text = ToolParser.convert_tool_calls_to_text(msg["tool_calls"])
                content = msg.get("content") or ""
                processed[i]["content"] = f"{content}\n{tool_calls_text}".strip()
                del processed[i]["tool_calls"]
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                processed[i]["role"] = "user"
                processed[i]["content"] = ToolParser.convert_tool_result_to_text(tool_call_id, msg.get("content", ""))
                processed[i].pop("tool_call_id", None)
        if tools:
            processed = self._add_tool_instructions(processed, tools)
        return processed

    def _add_tool_instructions(self, messages: List[Dict], tools: List[Dict]) -> List[Dict]:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"]
        instruction = f"\n[TOOL CALLING]\nWhen calling a tool, use JSON:\n```json\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n```\nAvailable: {', '.join(tool_names)}"
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["content"] = str(messages[i].get("content", "")) + instruction
                break
        return messages

    def _cleanup_chat(self, chat_id: str):
        if chat_id in self._active_chats:
            del self._active_chats[chat_id]
        try:
            self.adapter.delete_chat(chat_id)
        except Exception:
            pass

    def create_chat(self, model: str, title: str = 'New Chat') -> str:
        return self.adapter.create_chat(self.adapter.map_model(model), title)

    def delete_chat(self, chat_id: str) -> bool:
        if chat_id in self._active_chats:
            del self._active_chats[chat_id]
        return self.adapter.delete_chat(chat_id)

    def delete_all_chats(self) -> bool:
        self._active_chats.clear()
        return self.adapter.delete_all_chats()
