"""训练 runner 占位，封装训练流程入口（参考 LLaMA-Factory 的 run_exp）。"""
from typing import Optional, Dict, Any


def run_exp(args: Optional[Dict[str, Any]] = None) -> None:
    """占位实现：打印收到的 args。后续在此集成 HF/accelerate/peft 等训练逻辑。"""
    print("run_exp called with:", args)
