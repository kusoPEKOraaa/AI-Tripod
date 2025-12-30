-- SQLite schema for AI-Tripod (ModelVerse)
-- Generated from `modelverse/database.py:init_db()`
-- Usage:
--   sqlite3 modelverse/modelverse.db < commit/db/sqlite_schema.sql

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL,
  display_name TEXT DEFAULT '',
  phone TEXT DEFAULT '',
  is_admin BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  allowed_gpu_ids TEXT,
  allowed_task_types TEXT
);

CREATE TABLE IF NOT EXISTS resources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT,
  repo_id TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  progress REAL DEFAULT 0.0,
  size_mb REAL,
  local_path TEXT,
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS training_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  base_model_id INTEGER NOT NULL,
  dataset_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  progress REAL DEFAULT 0.0,
  config_params TEXT,
  config_path TEXT,
  output_path TEXT,
  output_model_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  error_message TEXT,
  FOREIGN KEY (user_id) REFERENCES users (id),
  FOREIGN KEY (base_model_id) REFERENCES resources (id),
  FOREIGN KEY (dataset_id) REFERENCES resources (id)
);

CREATE TABLE IF NOT EXISTS training_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  content TEXT NOT NULL,
  level TEXT DEFAULT 'INFO',
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (task_id) REFERENCES training_tasks (id)
);

CREATE TABLE IF NOT EXISTS inference_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  model_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  port INTEGER,
  api_base TEXT,
  process_id INTEGER,
  gpu_memory REAL,
  share_enabled BOOLEAN DEFAULT FALSE,
  display_name TEXT,
  tensor_parallel_size INTEGER DEFAULT 1,
  max_model_len INTEGER DEFAULT 4096,
  quantization TEXT,
  dtype TEXT DEFAULT 'auto',
  max_tokens INTEGER DEFAULT 2048,
  temperature REAL DEFAULT 0.7,
  top_p REAL DEFAULT 0.9,
  top_k INTEGER DEFAULT 50,
  repetition_penalty REAL DEFAULT 1.1,
  presence_penalty REAL DEFAULT 0.0,
  frequency_penalty REAL DEFAULT 0.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  stopped_at TIMESTAMP,
  error_message TEXT,
  FOREIGN KEY (user_id) REFERENCES users (id),
  FOREIGN KEY (model_id) REFERENCES resources (id)
);

CREATE TABLE IF NOT EXISTS evaluation_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  model_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  benchmark_type TEXT NOT NULL,
  status TEXT NOT NULL,
  progress REAL DEFAULT 0.0,
  num_fewshot INTEGER DEFAULT 0,
  custom_dataset_path TEXT,
  metrics TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  error_message TEXT,
  result_path TEXT,
  FOREIGN KEY (user_id) REFERENCES users (id),
  FOREIGN KEY (model_id) REFERENCES resources (id)
);

CREATE TABLE IF NOT EXISTS evaluation_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  content TEXT NOT NULL,
  level TEXT DEFAULT 'INFO',
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (task_id) REFERENCES evaluation_tasks (id)
);

CREATE TABLE IF NOT EXISTS active_downloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  resource_id INTEGER NOT NULL,
  pid INTEGER,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (resource_id) REFERENCES resources (id)
);

CREATE TABLE IF NOT EXISTS captchas (
  id TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

