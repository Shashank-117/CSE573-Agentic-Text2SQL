"""Shared LLM inference interface (single-agent and two-agent systems)."""

from .client import (
    Generation,
    GenerationParams,
    LLMClient,
    LLMError,
    Message,
    OpenAICompatibleClient,
    ScriptedLLM,
)
from .gpu import GpuPeakSampler, gpu_memory_used_mb

__all__ = [
    "Generation", "GenerationParams", "LLMClient", "LLMError", "Message",
    "OpenAICompatibleClient", "ScriptedLLM", "GpuPeakSampler", "gpu_memory_used_mb",
]
