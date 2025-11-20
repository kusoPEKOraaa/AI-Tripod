# 文件概览 — AI_Tripod_backend

下列文件为本仓库中已创建的骨架（scaffold），每个文件的作用与建议：

根目录
- `README.md`：项目总体说明与快速启动示例（如何安装依赖、启动 API、Web UI、训练入口）。
- `requirements.txt`：最小依赖清单（包含 Django、DRF、uvicorn、gradio；并标注了可选的 ML 库）。
- `manage.py`：Django 的管理入口脚本（用于运行管理命令、migrate、runserver 等）。
- `run_webui.py`：独立脚本，用于启动一个简单的 Gradio demo，直接调用 `ai_tripod_backend.chat.ChatModel`。
- `run_train.py`：训练入口脚本，会尝试通过 Django management command (`run_train`) 启动训练；若 Django 不可用则直接调用 `ai_tripod_backend.train.runner.run_exp`。

项目包：`ai_tripod_backend/`
- `__init__.py`：包初始化。
- `asgi.py`：ASGI entrypoint，供 `uvicorn` 启动（`uvicorn ai_tripod_backend.asgi:application`）。
- `wsgi.py`：WSGI entrypoint，供传统 WSGI 服务器（如 gunicorn）使用。
- `settings.py`：Django 配置（开发模式默认值），包括已安装 app 列表（`api`、`train`、`webui`）和 sqlite3 数据库配置。
- `urls.py`：项目级路由，已注册 `/api/` 指向 `ai_tripod_backend.api` 子应用。
- `chat.py`：ChatModel 抽象（占位实现）。该类用于统一模型加载/推理接口，方便 API 与 WebUI 调用。当前实现为回显示例，后续可替换为真实推理引擎（transformers、vLLM、modelscope 等）。

子应用：`ai_tripod_backend/api/`
- `__init__.py`：子应用初始化。
- `apps.py`：Django AppConfig。
- `serializers.py`：DRF 风格的请求/响应序列化器（`ChatRequestSerializer`、`ChatResponseSerializer`）。
- `views.py`：包含 `ChatCompletionView`，用于处理 `/api/chat/completions` POST 请求，调用 `ChatModel.generate`。当前不支持流式返回（返回 501）。
- `urls.py`：子应用路由（`chat/completions`）。

子应用：`ai_tripod_backend/train/`
- `__init__.py`、`apps.py`：Django app 注册。
- `runner.py`：训练 runner 占位（`run_exp(args)`），用于封装训练流程（参考 LLaMA-Factory 的 `run_exp`）。
- `management/commands/run_train.py`：Django 管理命令样例，直接调用 `run_exp`。

子应用：`ai_tripod_backend/webui/`
- `__init__.py`、`apps.py`：Web UI app 注册。
- Web UI 的实际 demo 由根目录 `run_webui.py` 启动；后续可把 Gradio 集成到该 app 下并通过 Django 路由提供 WebUI 页面。

Notes（备注）
- 本仓库是一个最小的骨架，旨在尽快搭起团队协作的开发框架。大多数 ML/推理功能（模型加载、量化、训练细节、流式 SSE）都以占位/说明形式提供，便于按需实现。
- 代码中出现的 lint 报告（例如 "无法解析导入 django"）通常是因为运行环境尚未安装依赖，不影响文件内容本身。
