"""LLM backends behind one tiny interface: complete(system, user) -> str.

Two implementations:
  * AnthropicLLM : hosted Claude via the Anthropic API (best answer quality / citation discipline)
  * HFLLM        : any local Hugging Face chat model (free, private, needs a GPU for 3B+)

Adding another provider (OpenAI, Ollama, vLLM ...) means subclassing LLM and implementing one method.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .config import Config


class LLM(ABC):
    @abstractmethod
    def complete(
        self, system: str, user: str, max_tokens: int = 700, temperature: float | None = 0.0
    ) -> str: ...


class AnthropicLLM(LLM):
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env or export it).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, system, user, max_tokens=700, temperature=0.0) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


class HFLLM(LLM):
    def __init__(self, model_name: str):
        import torch  # noqa: F401  (ensures a clear import error if torch is missing)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        print(f"--- Model is currently running on device: {self.model.device} ---")

    def complete(self, system, user, max_tokens=700, temperature=0.0) -> str:
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        gen = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()


def get_llm(cfg: Config) -> LLM:
    provider = cfg.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicLLM(cfg.anthropic_model)
    if provider == "hf":
        return HFLLM(cfg.hf_llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER {cfg.llm_provider!r} (use 'anthropic' or 'hf')")
