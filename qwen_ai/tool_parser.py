"""Tool Parser for Qwen AI - Optimized for Reliability"""

import json
import re
import uuid
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ToolParser:
    BACKTICK3 = '```'
    TOOL_CALL_PREFIX = "call_"

    @staticmethod
    def build_tool_system_prompt(tools: List[Dict], tool_choice: Optional[Any] = None) -> str:
        if not tools:
            return ""
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                tool_descriptions.append(f"- {name}: {desc}")
        if tool_descriptions:
            return f"Available tools:\n" + "\n".join(tool_descriptions)
        return ""

    @classmethod
    def convert_tool_calls_to_text(cls, tool_calls: List[Dict]) -> str:
        if not tool_calls:
            return ""
        parts = []
        for tc in tool_calls:
            func = tc.get("function", {})
            if not func:
                continue
            name = func.get("name", "")
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                except json.JSONDecodeError:
                    args_dict = {}
            elif isinstance(args, dict):
                args_dict = args
            else:
                args_dict = {}
            call_dict = {"name": name, "arguments": args_dict}
            json_str = json.dumps(call_dict, ensure_ascii=False, indent=2)
            parts.append(f"{cls.BACKTICK3}json\n{json_str}\n{cls.BACKTICK3}")
        return "\n\n".join(parts)

    @staticmethod
    def convert_tool_result_to_text(tool_call_id: str, content: str) -> str:
        if not content:
            return f"Tool Result [{tool_call_id}]: No output\n"
        return f"Tool Result [{tool_call_id}]:\n{content}\n"

    @classmethod
    def extract_tool_calls(cls, text: str) -> Tuple[str, List[Dict]]:
        if not text or not text.strip():
            return "", []
        tool_calls = []
        positions_to_remove = []
        cls._extract_from_code_blocks(text, tool_calls, positions_to_remove)
        cls._extract_from_chinese_brackets(text, tool_calls, positions_to_remove)
        cls._extract_from_tool_call_tags(text, tool_calls, positions_to_remove)
        cls._extract_raw_json(text, tool_calls, positions_to_remove)
        clean_text = text
        for start, end in sorted(positions_to_remove, key=lambda x: x[0], reverse=True):
            clean_text = clean_text[:start] + clean_text[end:]
        clean_text = cls._cleanup_text(clean_text)
        return clean_text, tool_calls

    @classmethod
    def _extract_from_code_blocks(cls, text: str, tool_calls: List[Dict], positions: List):
        pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(text):
            json_str = match.group(1).strip()
            parsed = cls._parse_tool_json(json_str)
            if parsed:
                tool_calls.extend(parsed)
                positions.append((match.start(), match.end()))

    @classmethod
    def _extract_from_chinese_brackets(cls, text: str, tool_calls: List[Dict], positions: List):
        pattern = re.compile(r'【\s*(.*?)\s*】', re.DOTALL)
        for match in pattern.finditer(text):
            content = match.group(1).strip()
            parsed = cls._parse_tool_json(content)
            if parsed:
                tool_calls.extend(parsed)
                positions.append((match.start(), match.end()))

    @classmethod
    def _extract_from_tool_call_tags(cls, text: str, tool_calls: List[Dict], positions: List):
        pattern = re.compile(r'\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]', re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(text):
            content = match.group(1).strip()
            parsed = cls._parse_tool_json(content)
            if parsed:
                tool_calls.extend(parsed)
                positions.append((match.start(), match.end()))

    @classmethod
    def _extract_raw_json(cls, text: str, tool_calls: List[Dict], positions: List):
        pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+)"[^{}]*"arguments"\s*:\s*(?:\{[^{}]*\}|"[^"]*")[^{}]*\}', re.DOTALL)
        for match in pattern.finditer(text):
            json_str = match.group(0)
            parsed = cls._parse_tool_json(json_str)
            if parsed:
                for p in parsed:
                    if not cls._tool_call_exists(tool_calls, p):
                        tool_calls.append(p)
                        positions.append((match.start(), match.end()))

    @classmethod
    def _parse_tool_json(cls, json_str: str) -> List[Dict]:
        tool_calls = []
        try:
            parsed = json.loads(json_str)
            return cls._extract_tools_from_parsed(parsed)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
        fixed_json = cls._fix_json_string(json_str)
        try:
            parsed = json.loads(fixed_json)
            return cls._extract_tools_from_parsed(parsed)
        except json.JSONDecodeError as e:
            logger.debug(f"Fixed JSON parse also failed: {e}")
        return cls._extract_tools_regex(json_str)

    @classmethod
    def _extract_tools_from_parsed(cls, parsed: Any) -> List[Dict]:
        tool_calls = []
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = item.get("name", "")
            if not name or not isinstance(name, str):
                continue
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', str(name).strip())
            if not clean_name:
                continue
            args = item.get("arguments", {})
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            elif isinstance(args, str):
                try:
                    json.loads(args)
                    args_str = args
                except json.JSONDecodeError:
                    args_str = "{}"
            else:
                args_str = "{}"
            tool_calls.append({
                "id": f"{cls.TOOL_CALL_PREFIX}{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": clean_name, "arguments": args_str}
            })
        return tool_calls

    @classmethod
    def _extract_tools_regex(cls, text: str) -> List[Dict]:
        tool_calls = []
        name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
        args_pattern = re.compile(r'"arguments"\s*:\s*(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*")')
        name_match = name_pattern.search(text)
        args_match = args_pattern.search(text)
        if name_match:
            name = name_match.group(1).strip()
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
            if clean_name:
                args_str = "{}"
                if args_match:
                    try:
                        json.loads(args_match.group(1))
                        args_str = args_match.group(1)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append({
                    "id": f"{cls.TOOL_CALL_PREFIX}{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": clean_name, "arguments": args_str}
                })
        return tool_calls

    @staticmethod
    def _fix_json_string(json_str: str) -> str:
        fixed = json_str
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', fixed)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        fixed = fixed.replace("'", '"')
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        if open_braces > 0:
            fixed += '}' * open_braces
        if open_brackets > 0:
            fixed += ']' * open_brackets
        return fixed

    @staticmethod
    def _tool_call_exists(tool_calls: List[Dict], new_call: Dict) -> bool:
        new_name = new_call.get("function", {}).get("name", "")
        for existing in tool_calls:
            if existing.get("function", {}).get("name") == new_name:
                return True
        return False

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = re.sub(r'</?tool_call>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?arg_value>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def has_tool_use(cls, content: str) -> bool:
        if not content:
            return False
        patterns = [
            r'```(?:json)?\s*\n?\{',
            r'【\s*\{',
            r'\[TOOL_CALL\]',
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"',
        ]
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                return True
        return False
