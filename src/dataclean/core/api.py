#!/usr/bin/env python3
"""
core/api.py — 统一 DashScope API 客户端

封装 OpenAI 兼容 API 调用，供文本 QC 和视觉 QC 脚本共用。
内置重试、自适应限流、速率控制。

用法:
    from dataclean.core.api import DashScopeClient

    client = DashScopeClient(model="qwen3.5-flash")
    response = client.chat("系统提示词", "用户问题")
    # 或带图片
    response = client.chat_with_image("描述这张图", image_base64=..., content_type="image/jpeg")
"""

from __future__ import annotations

import os
import time
import random
from typing import Any
from dataclasses import dataclass, field


@dataclass
class ApiConfig:
    """API 配置。

    从环境变量加载，可通过参数覆盖。
    """
    api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.5-flash"
    timeout: int = 60
    max_retries: int = 3
    backoff_base: float = 2.0
    jitter_max: float = 0.5


class DashScopeClient:
    """DashScope API 客户端（OpenAI 兼容接口）。

    特性:
      - 自动重试 + 指数退避 + 随机抖动
      - 可选集成 AdaptiveConcurrencyGate（API 429 自适应）
      - 支持纯文本和图片输入

    用法:
        client = DashScopeClient(model="qwen-vl-flash")
        resp = client.chat("你是审校助手。", "该视频是否属于影视原片？仅输出 T 或 F。")
        print(resp)  # "T"
    """

    def __init__(self, config: ApiConfig | None = None, **overrides):
        """
        Args:
            config: ApiConfig 对象，None 则使用默认值（从环境变量加载）
            **overrides: 覆盖 config 中的任意字段
        """
        if config is None:
            config = ApiConfig()
        self._api_key = overrides.pop("api_key", None) or config.api_key
        self._api_base = overrides.pop("api_base", None) or config.api_base
        self._model = overrides.pop("model", None) or config.model
        self._timeout = overrides.pop("timeout", None) or config.timeout
        self._max_retries = overrides.pop("max_retries", None) or config.max_retries
        self._backoff_base = overrides.pop("backoff_base", None) or config.backoff_base
        self._jitter_max = overrides.pop("jitter_max", None) or config.jitter_max

        self._client = None
        self._gate = None   # AdaptiveConcurrencyGate（可选）

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):
        """延迟创建 OpenAI 客户端。"""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要 openai 包: pip install openai")
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
            timeout=self._timeout,
        )

    def set_concurrency_gate(self, gate) -> None:
        """设置 AdaptiveConcurrencyGate，用于 429 自适应限流。

        gate 需要有 acquire() / release() / on_rate_limit() 方法。
        """
        self._gate = gate

    # ── 核心 API 调用 ──

    def chat(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """发送纯文本聊天请求，返回响应文本。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户消息
            **kwargs: 覆盖参数 (temperature, max_tokens 等)

        Returns:
            API 响应文本（已 strip）

        Raises:
            RuntimeError: 所有重试后仍失败
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_with_retry(messages, **kwargs)

    def chat_with_image(
        self,
        user_prompt: str,
        image_base64: str,
        content_type: str = "image/jpeg",
        system_prompt: str = "",
        **kwargs,
    ) -> str:
        """发送图文聊天请求（视觉模型），返回响应文本。

        Args:
            user_prompt: 用户文本提示
            image_base64: Base64 编码的图片
            content_type: 图片 MIME 类型
            system_prompt: 可选的系统提示词
            **kwargs: 覆盖参数

        Returns:
            API 响应文本（已 strip）
        """
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{image_base64}"}},
            {"type": "text", "text": user_prompt},
        ]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self._call_with_retry(messages, **kwargs)

    def _call_with_retry(self, messages: list[dict], **kwargs) -> str:
        """带重试的核心调用。"""
        self._ensure_client()

        temperature = kwargs.pop("temperature", 0.1)
        max_tokens = kwargs.pop("max_tokens", 5)

        last_error = None
        for attempt in range(self._max_retries + 1):
            # 自适应并发控制
            if self._gate:
                self._gate.acquire()

            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # 429 降级
                if "429" in error_str or "rate" in error_str:
                    if self._gate:
                        self._gate.on_rate_limit()
                    if attempt < self._max_retries:
                        delay = self._backoff_base ** (attempt + 1) + random.uniform(0, self._jitter_max)
                        time.sleep(delay)
                        continue
                # 其他错误
                elif attempt < self._max_retries:
                    time.sleep(self._backoff_base ** (attempt + 1))
                    continue
            finally:
                if self._gate:
                    self._gate.release()

        raise RuntimeError(f"API 调用失败（{self._max_retries + 1} 次重试后）: {last_error}")


# ── 便捷工厂 ──

def create_client(model: str | None = None, category: str = "default", **overrides) -> DashScopeClient:
    """创建预配置的 DashScope 客户端。

    Args:
        model: 模型名（None 则用默认）
        category: 类别名（未来可根据类别选不同模型）
        **overrides: 其他覆盖参数

    Returns:
        配置好的 DashScopeClient
    """
    config = ApiConfig()
    if model:
        config.model = model
    return DashScopeClient(config=config, **overrides)
