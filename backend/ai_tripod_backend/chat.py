"""ChatModel 抽象，用于统一模型接口（轻量占位）。"""
from enum import Enum
from typing import Any, Dict


class EngineName(str, Enum):
    HF = "hf"
    MOCK = "mock"


class ChatModel:
    """Minimal ChatModel abstraction.

    方法:
        - generate(prompt, **kwargs): 返回 dict 结果
    """

    def __init__(self, engine: EngineName = EngineName.MOCK, model_name: str | None = None):
        self.engine = engine
        self.model_name = model_name or "mock-model"
        # can_generate 用于 API 层判断是否可做生成
        self.can_generate = True if engine != EngineName.MOCK else True

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """占位实现：回显 prompt，实际应由 transformers/vllm 等替换。"""
        # TODO: integrate with HF transformers or other engines
        return {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": prompt[::-1]}}
            ],
        }
