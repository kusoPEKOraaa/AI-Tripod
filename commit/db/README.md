# 数据库脚本与样例数据

本项目默认使用 SQLite，首次启动 `modelverse/main.py` 时会在 `modelverse/modelverse.db` 自动建表并创建默认管理员账号。

## 1) 建表脚本（SQLite）

- `commit/db/sqlite_schema.sql`：建表脚本（来自代码中 `init_db()` 的表结构整理版）。

手动初始化（可选）：

```bash
sqlite3 modelverse/modelverse.db < commit/db/sqlite_schema.sql
```

## 2) 样例数据

- `commit/db/sample_export.json`：小规模样例数据（用于作业提交展示，不影响系统运行）。

