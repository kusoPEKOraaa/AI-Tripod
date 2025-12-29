"""
临时修复脚本 - 检查并修复已下载但显示失败的资源
"""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "modelverse.db"

def fix_downloaded_resources():
    """检查并修复已下载的资源"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查询所有FAILED或DOWNLOADING状态的资源
    cursor.execute("""
        SELECT id, name, repo_id, local_path, status, progress
        FROM resources
        WHERE status IN ('FAILED', 'DOWNLOADING')
        AND local_path IS NOT NULL
    """)

    resources = cursor.fetchall()
    print(f"找到 {len(resources)} 个需要修复的资源")

    fixed_count = 0
    for resource_id, name, repo_id, local_path, status, progress in resources:
        print(f"\n检查资源: {name}")
        print(f"  本地路径: {local_path}")

        # 检查本地路径是否存在
        if not local_path:
            print(f"  ✗ 本地路径为空，跳过")
            continue

        save_dir = Path(local_path)
        if not save_dir.exists():
            print(f"  ✗ 目录不存在，跳过")
            continue

        # 递归扫描所有文件
        all_files = []
        for dirpath, dirnames, filenames in os.walk(save_dir):
            for filename in filenames:
                if not filename.startswith('.'):
                    all_files.append(os.path.join(dirpath, filename))

        print(f"  扫描到 {len(all_files)} 个文件")

        # 检查关键文件
        key_files = ['model.safetensors', 'pytorch_model.bin', 'config.json', 'tokenizer.json']
        has_key_files = any(any(os.path.basename(f).endswith(kf) for kf in key_files) for f in all_files)

        # 如果有文件且包含关键文件，标记为完成
        if len(all_files) > 0 and (has_key_files or len(all_files) >= 5):
            print(f"  ✓ 资源有效，包含关键文件: {has_key_files}")

            # 更新数据库
            cursor.execute("""
                UPDATE resources
                SET status = 'COMPLETED',
                    progress = 100.0
                WHERE id = ?
            """, (resource_id,))
            fixed_count += 1
            print(f"  ✓ 已更新为完成状态")
        else:
            print(f"  ✗ 资源无效，跳过")

    # 提交更改
    conn.commit()
    conn.close()

    print(f"\n修复完成！共修复 {fixed_count} 个资源")

if __name__ == "__main__":
    fix_downloaded_resources()
