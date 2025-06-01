# ─── ai21_wrapper.py ────────────────────────────────────────────────────────────
# pip install ai21
from __future__ import annotations
import logging
from functools import cached_property
from typing import Any, ClassVar, List, Optional
from pydantic import Field, PrivateAttr

import numpy as np
from ai21 import AI21Client            # official SDK ≥ 3.2.0
from ai21.models.chat import ChatMessage, ResponseFormat


from ..logging import beam_logger as logger
from ..path import beam_key
from .core import BeamLLM, CompletionObject


class AI21LLM(BeamLLM):
    """
    Generic AI21 Labs wrapper that exposes a superset of the Jamba-chat
    and Jurassic-2 completion arguments, model listing, usage tracking, etc.
    """

    # --------------------------------------------------------------------- init
    api_key: Optional[str] = Field(None)
    api_base: Optional[str] = Field("https://api.ai21.com/v1")
    _models: Any = PrivateAttr(default=None)

    # Supported kwargs (taken from official docs / SDK)  :contentReference[oaicite:0]{index=0}
    chat_kwargs: ClassVar[List[str]] = [
        "max_tokens", "temperature", "top_p", "stop", "n", "stream",
        "documents",  # RAG beta
        "tools", "tool_choice", "response_format", "seed",
    ]
    completion_kwargs: ClassVar[List[str]] = [
        "max_tokens", "temperature", "top_p", "top_k", "presence_penalty",
        "frequency_penalty", "stop", "num_results", "stream",
    ]

    # ------------------------------------------------------------------ ctor
    def __init__(
        self,
        model: str = None,
        api_key: str | None = None,
        hostname: str | None = None,
        *args,
        **kwargs,
    ):

        kwargs["scheme"] = "ai21"
        super().__init__(*args, model=model, **kwargs)

        self.api_key = api_key or beam_key("AI21_API_KEY", api_key)
        if hostname:
            self.api_base = hostname

    # A naive heuristic – tweak if you add other AI21 families
    @property
    def is_chat(self) -> bool:
        return self.model.startswith("jamba")


    # -------------------------------------------------------------- low-level client
    @cached_property
    def client(self) -> AI21Client:
        return AI21Client(api_key=self.api_key, base_url=self.api_base)

    # -------------------------------------------------------------- usage helpers
    def update_usage(self, response):
        if "usage" in response:
            usage = response["usage"]
            self.usage["prompt_tokens"] += usage["prompt_tokens"]
            self.usage["completion_tokens"] += usage["completion_tokens"]
            self.usage["total_tokens"] += usage["total_tokens"]

    # -------------------------------------------------------------- completions
    def _completion(self, prompt: str, **kwargs):
        kwargs = self.filter_completion_kwargs(kwargs)
        res = self.client.completions.create(model=self.model, prompt=prompt, **kwargs)
        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def _chat_completion(self, chat, stream: bool | None = None, guidance=None, **kwargs):

        response_format = ResponseFormat(type="text")
        if guidance is not None:
            response_format = ResponseFormat(type="json")

        kwargs = self.filter_chat_kwargs(kwargs)
        messages_in = self._convert_messages(chat.openai_format)
        res = self.client.chat.completions.create(
            model=self.model, messages=messages_in, stream=stream, response_format=response_format, **kwargs
        )
        return CompletionObject(prompt=messages_in, kwargs=kwargs, response=res)

    # -------------------------------------------------------------- utils
    @staticmethod
    def filter_chat_kwargs(kwargs):
        return {k: v for k, v in kwargs.items() if k in AI21LLM.chat_kwargs and v is not None}

    @staticmethod
    def filter_completion_kwargs(kwargs):
        return {
            k: v for k, v in kwargs.items() if k in AI21LLM.completion_kwargs and v is not None
        }

    def verify_response(self, res):
        stream = res.stream
        res = res.response
        if not hasattr(res.choices[0], "finish_reason"):
            return False
        if res.choices[0].finish_reason != "stop" and not stream:
            logger.warning(f"finish_reason={res.choices[0].finish_reason}")
        return True

    def extract_text(self, res):
        stream = res.stream
        res = res.response

        if not self.is_chat:
            return res.choices[0].text
        if stream:
            return res.choices[0].delta.content
        return res.choices[0].message.content

    def parse_json(self, res):
        if not self.is_chat:
            return None
        msg = res.response.choices[0].message
        return getattr(msg, "parsed", None)

    def ai21_format(self, res):
        return res.response

    # -------------------------------------------------------------- misc helpers
    def retrieve(self, model: str | None = None):
        # Not exposed in AI21; raise or return None
        raise NotImplementedError("AI21 does not expose `Engine.retrieve`.")

    @property
    def models(self):
        if self._models is None:
            # SDK exposes .models.list()
            self._models = {m.id: m for m in self.client.models.list().data}
        return self._models

    # -------------------------------------------------------------- embeddings
    def embedding(self, text: str, model: str | None = None):
        raise NotImplementedError("AI21 currently has no public embedding endpoint.")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _convert_messages(msgs):
        """
        Beam's `openai_format` already matches the shape
        {'role': str, 'content': str, ...}. Convert to sdk ChatMessage list.
        """
        out = []
        for m in msgs:
            # allow passthrough if user already supplied ChatMessage
            if isinstance(m, ChatMessage):
                out.append(m)
            else:
                out.append(ChatMessage(role=m["role"], content=m["content"]))
        return out

