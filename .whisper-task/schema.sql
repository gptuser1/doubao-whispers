-- 豆包悄悄话回复表
CREATE TABLE IF NOT EXISTS replies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  whisper_id TEXT NOT NULL,
  nickname TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  reply_to TEXT DEFAULT '',
  reply_to_floor INTEGER DEFAULT NULL,
  floor INTEGER NOT NULL,
  ip_hash TEXT NOT NULL,
  is_doubao INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 按 whisper_id 查询的索引
CREATE INDEX IF NOT EXISTS idx_replies_whisper_id ON replies (whisper_id);

-- 按 whisper_id + timestamp 排序的索引
CREATE INDEX IF NOT EXISTS idx_replies_whisper_id_timestamp ON replies (whisper_id, timestamp);

-- 频率限制用的索引
CREATE INDEX IF NOT EXISTS idx_replies_ip_hash ON replies (ip_hash, whisper_id, created_at);
