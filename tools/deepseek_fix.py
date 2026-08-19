# tools/deepseek_fix.py
"""
DeepSeek thinking 模式兼容修复。

问题背景（官方 issue: langchain-ai/langchain#37178）：
  DeepSeek V4 系列（如 deepseek-v4-flash）开启 thinking 模式后，API 会额外返回
  `reasoning_content` 字段。当 function calling 需要把工具执行结果回传给模型时，
  API 要求将上一轮的 `reasoning_content` **原样**一并传回，否则返回 400：
    "The `reasoning_content` in the thinking mode must be passed back to the API."

  当前 langchain-deepseek 尚未对该字段做兼容处理（第二轮请求会缺失它）。

方案：
  继承 ChatDeepSeek，重写 `_get_request_payload`，从原始消息的
  `additional_kwargs["reasoning_content"]` 取出思考内容，补回发给 API 的
  assistant 消息中。同时修复 tool 消息 content 为 JSON 字符串等兼容细节。

使用：
  from tools.deepseek_fix import ChatDeepSeekFixReasoningContent
  llm = ChatDeepSeekFixReasoningContent(
      model="deepseek-v4-flash",
      api_key=...,
      base_url=...,
      temperature=0,
  )
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek


class ChatDeepSeekFixReasoningContent(ChatDeepSeek):
    """在 DeepSeek thinking 模式下，正确回传 reasoning_content 的 ChatDeepSeek。"""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        # 调用 BaseChatOpenAI._get_request_payload 构造基础 payload
        payload = super(ChatDeepSeek, self)._get_request_payload(input_, stop=stop, **kwargs)
        # 转换输入消息，用于读取 additional_kwargs 中的 reasoning_content
        input_messages = self._convert_input(input_).to_messages() or []

        for idx, message in enumerate(payload.get("messages", [])):
            reasoning_content = (
                input_messages[idx].additional_kwargs.get("reasoning_content")
                if idx < len(input_messages)
                else None
            )
            # 把上一轮 assistant 的 reasoning_content 原样补回 payload
            if reasoning_content and message.get("role") == "assistant":
                message["reasoning_content"] = reasoning_content
            # DeepSeek 兼容：tool 消息 content 需为 JSON 字符串；assistant 消息 content 需为纯文本
            if message.get("role") == "tool" and isinstance(message.get("content"), list):
                message["content"] = json.dumps(message["content"], ensure_ascii=False)
            elif message.get("role") == "assistant" and isinstance(message.get("content"), list):
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
        return payload
