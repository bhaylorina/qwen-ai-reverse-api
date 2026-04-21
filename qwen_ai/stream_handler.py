"""Stream Handler for Qwen AI"""

import json
import time
import uuid
import logging
from typing import Optional, Callable, Dict, Any, Generator, List
from http.client import IncompleteRead
from .tool_parser import ToolParser

logger = logging.getLogger(__name__)


class QwenAiStreamHandler:
    def __init__(self, model: str, on_end: Optional[Callable[[str], None]] = None):
        self.chat_id = ''
        self.model = model
        self.created = int(time.time())
        self.on_end = on_end
        self.response_id = ''
        self.content = ''
        self.tool_calls_sent = False
        self.initial_chunk_sent = False

    def set_chat_id(self, chat_id: str):
        self.chat_id = chat_id

    def _make_chunk(self, delta: Dict, finish_reason: Optional[str] = None) -> str:
        resp_id = self.response_id or f"chatcmpl-{self.chat_id}"
        chunk = {"id": resp_id, "model": self.model, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], "created": self.created}
        return f"data: {json.dumps(chunk)}\n\n"

    def handle_stream(self, response) -> Generator[str, None, None]:
        reasoning_text = ''
        has_sent_reasoning = False
        try:
            for line in response.iter_lines():
                if not line or not line.decode('utf-8').startswith('data: '):
                    continue
                data_str = line.decode('utf-8')[6:]
                if data_str == '[DONE]':
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if data.get('response.created', {}).get('response_id'):
                    self.response_id = data['response.created']['response_id']
                if not data.get('choices'):
                    continue
                delta = data['choices'][0].get('delta', {})
                phase = delta.get('phase')
                status = delta.get('status')
                content = delta.get('content', '')
                if phase == 'think' and status != 'finished':
                    reasoning_text += content
                    if not has_sent_reasoning:
                        yield self._make_chunk({"role": "assistant", "reasoning_content": ""})
                        has_sent_reasoning = True
                    if content:
                        yield self._make_chunk({"reasoning_content": content})
                elif phase == 'answer' or (phase is None and content):
                    if content:
                        self.content += content
                        yield self._make_chunk({"content": content})
                if status == 'finished' and (phase == 'answer' or phase is None):
                    clean_text, tool_calls = ToolParser.extract_tool_calls(self.content)
                    if tool_calls:
                        for chunk in self._generate_tool_calls(tool_calls):
                            yield chunk
                    else:
                        yield self._make_chunk({}, "stop")
                        yield "data: [DONE]\n\n"
                    self._cleanup()
                    return
        except Exception as e:
            logger.error(f"Stream error: {e}")
        yield self._make_chunk({}, "stop")
        yield "data: [DONE]\n\n"
        self._cleanup()

    def handle_non_stream(self, response) -> Dict[str, Any]:
        result = {'id': '', 'model': self.model, 'object': 'chat.completion', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}, 'created': self.created}
        for line in response.iter_lines():
            if not line or not line.decode('utf-8').startswith('data: '):
                continue
            data_str = line.decode('utf-8')[6:]
            if data_str == '[DONE]':
                break
            try:
                parsed = json.loads(data_str)
                if parsed.get('response.created', {}).get('response_id'):
                    result['id'] = parsed['response.created']['response_id']
                if parsed.get('choices'):
                    delta = parsed['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        result['choices'][0]['message']['content'] += content
            except json.JSONDecodeError:
                continue
        clean_text, tool_calls = ToolParser.extract_tool_calls(result['choices'][0]['message']['content'])
        if tool_calls:
            result['choices'][0]['message']['content'] = clean_text or None
            result['choices'][0]['message']['tool_calls'] = tool_calls
            result['choices'][0]['finish_reason'] = 'tool_calls'
        self._cleanup()
        return result

    def _generate_tool_calls(self, tool_calls: List[Dict]) -> Generator[str, None, None]:
        self.tool_calls_sent = True
        resp_id = self.response_id or f"chatcmpl-{uuid.uuid4().hex[:16]}"
        if not self.initial_chunk_sent:
            yield self._make_chunk({"role": "assistant", "content": ""})
            self.initial_chunk_sent = True
        for i, tc in enumerate(tool_calls):
            chunk1 = {"id": resp_id, "model": self.model, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}]}, "finish_reason": None}], "created": self.created}
            yield f"data: {json.dumps(chunk1)}\n\n"
            chunk2 = {"id": resp_id, "model": self.model, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}], "created": self.created}
            yield f"data: {json.dumps(chunk2)}\n\n"
        yield self._make_chunk({}, "tool_calls")
        yield "data: [DONE]\n\n"
        self._cleanup()

    def _cleanup(self):
        if self.on_end and self.chat_id:
            try:
                self.on_end(self.chat_id)
            except Exception:
                pass
