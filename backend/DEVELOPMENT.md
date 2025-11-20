# 开发文档 — AI_Tripod_backend 扩展与二次开发指南

此文档旨在指导团队如何在现有骨架上继续开发（接入真实模型、添加鉴权、实现流式输出、CI 配置等）。

1) 环境准备
- 推荐 Python 3.9+，创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 若需 GPU 与 transformers：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate safetensors
```

注意：`torch`、`vllm`、`modelscope` 等依赖在不同硬件/OS 上安装方法不同，请按官方文档安装。

2) 运行项目（开发）
- 初始化 DB（本 scaffold 使用 sqlite）：

```bash
python manage.py migrate
```

- 启动开发服务器：

```bash
python manage.py runserver 0.0.0.0:8000
# 或使用 ASGI
uvicorn ai_tripod_backend.asgi:application --host 0.0.0.0 --port 8000
```

- 启动 WebUI (Gradio demo)：

```bash
python run_webui.py
```

- 启动训练（scaffold）:

```bash
python run_train.py
# 或者使用 Django 管理命令
python manage.py run_train --config path/to/config.yaml
```

3) 把 `ChatModel` 与真实推理引擎集成
- 位置：`ai_tripod_backend/chat.py`。推荐实现以下 contract（接口规范）：

- 构造函数：ChatModel(engine=..., model_name=...)
- 属性：`engine`（标识具体引擎）、`can_generate`（是否支持生成）
- 方法：`generate(prompt: str, **kwargs) -> dict` 返回类似 OpenAI/ChatCompletion 的 dict（包含 choices/message 等）

示例（使用 transformers 的简化实现）：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFChatModel(ChatModel):
    def __init__(self, model_name: str):
        super().__init__(engine='hf', model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"id": "hf-1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}
```

注意事项：
- 考虑模型下载与缓存；对大模型建议使用 `safetensors` 与 `device_map="auto"`（或 accelerate 的 `init_empty_weights` + `load_checkpoint_and_dispatch`）。
- 推理时关注内存/显存管理（`torch.cuda.empty_cache()`、batching、半精度/量化）。

4) 在 API 中添加鉴权与限流
- 鉴权（简单）：使用 DRF 的 Permission，或在 `ai_tripod_backend/api/views.py` 中检查 header；推荐实现 `APIKeyPermission`：

```python
from rest_framework.permissions import BasePermission
import os

class APIKeyPermission(BasePermission):
    def has_permission(self, request, view):
        api_key = os.getenv('API_KEY')
        if not api_key:
            return True
        token = request.headers.get('Authorization', '')
        return token == f'Bearer {api_key}'
```

- 限流（推荐）：使用 Django 中间件 + Redis 实现（如 `django-ratelimit` 或 `drf-extensions`），在生产中也可在反向代理（Nginx、Cloudflare）层做限制。

5) 实现流式输出（SSE / WebSocket）
- 选项 A（轻量）：在视图中返回 Django StreamingHttpResponse，并逐步写入数据；但并发与长连接管理有限。
- 选项 B（推荐）：使用 Django Channels（ASGI）或单独用 FastAPI/Starlette 处理流式端点，然后在 Django 路由中反向代理到该服务。

示例（简化 SSE，注意生产要处理超时、并发）：

```python
from django.http import StreamingHttpResponse

def sse_stream(request):
    def event_stream():
        for chunk in generator():
            yield f"data: {chunk}\n\n"
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
```

6) 测试与 CI
- 编写单元测试（pytest + Django）：把 `ChatModel` 的实际推理部分用 mock 抽象，重点测试 API 的输入/输出格式与错误处理。示例目录 `tests/`。
- CI 建议（GitHub Actions）：
  - 步骤：checkout -> setup-python -> pip install -r requirements.txt（可选缩减 heavy deps） -> run ruff/flake8 -> run pytest（带 sqlite）。

7) 部署建议
- 推荐使用 ASGI（uvicorn/gunicorn + uvicorn workers，或 daphne）和反向代理（Nginx）。
- 若部署模型推理与 API 在同一台机器，注意：模型加载会占用大量内存/显存；建议把推理服务与轻量 API 分离，推理服务用更接近硬件的容器/节点。可考虑把推理服务独立为 microservice（FastAPI/vLLM），通过内部网络调用。
- Docker 化：为项目写一个轻量 Dockerfile（仅含 Django + uvicorn），并为模型推理写单独镜像（带 CUDA）。

8) 常见扩展与工程化要点（优先级建议）
- 优先：实现 API 鉴权、限流、错误处理与监控（日志、Prometheus metrics）。
- 次要：实现 ChatModel 的 HF/vLLM 适配层，支持模型热插拔与配置化加载。
- 长期：引入请求合并/批处理、推理缓存、多模型路由与自动伸缩（Kubernetes + HPA + GPU operator）。

9) 帮助与接入示例
- 如果希望我替你把 `ai_tripod_backend/chat.py` 实现为 `HFChatModel`（以 transformers 为后端）并把 `api/views.py` 改为使用共享单例模型加载，我可以继续实现并在本地做一次最小测试（非 GPU 情况下用小模型）。

——
如果你同意我现在继续实现 `ChatModel` 的 transformers 示例适配（并修改 API 使用单例加载），我将：
1) 在 `ai_tripod_backend/chat.py` 添加 `HFChatModel` 与工厂函数；
2) 在 `api/views.py` 中改为使用项目范围的 model singletons（懒加载）；
3) 运行基本的 smoke test（发起一个 POST 到 `/api/chat/completions` 并显示输出）。
