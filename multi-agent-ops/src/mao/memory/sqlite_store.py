"""SQLite-based persistent memory store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStore:
    """SQLite-backed store for tasks, messages, and metadata."""

    def __init__(self, db_path: str = "./data/mao.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                subtasks_json TEXT DEFAULT '[]',
                final_output TEXT,
                stats_json TEXT DEFAULT '{}',
                total_tokens INTEGER DEFAULT 0,
                rounds INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT DEFAULT '{}',
                parent_id TEXT,
                timestamp TEXT,
                priority INTEGER DEFAULT 3,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE TABLE IF NOT EXISTS context (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id);
        """)
        self._conn.commit()

    # ── Tasks ────────────────────────────────────────────────────────────

    def save_task(self, task: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, description, status, subtasks_json, final_output, stats_json,
                total_tokens, rounds, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task["id"],
                task.get("description", ""),
                task.get("status", "created"),
                json.dumps(task.get("subtasks", []), ensure_ascii=False, default=str),
                task.get("final_output"),
                json.dumps(task.get("stats", {}), ensure_ascii=False, default=str),
                task.get("total_tokens", 0),
                task.get("rounds", 0),
                str(task.get("created_at", "")),
                str(task.get("updated_at", "")),
            ),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "description": row["description"],
            "status": row["status"],
            "subtasks": json.loads(row["subtasks_json"]),
            "final_output": row["final_output"],
            "stats": json.loads(row["stats_json"]),
            "total_tokens": row["total_tokens"],
            "rounds": row["rounds"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "description": r["description"],
                "status": r["status"],
                "total_tokens": r["total_tokens"],
                "rounds": r["rounds"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    # ── Messages ─────────────────────────────────────────────────────────

    def save_message(self, task_id: str, message: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO messages
               (id, task_id, from_agent, to_agent, type, payload_json, parent_id, timestamp, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.get("id", ""),
                task_id,
                message.get("from_agent", ""),
                message.get("to_agent", ""),
                message.get("type", "task"),
                json.dumps(message.get("payload", {}), ensure_ascii=False, default=str),
                message.get("parent_id"),
                str(message.get("timestamp", "")),
                message.get("priority", 3),
            ),
        )
        self._conn.commit()

    def get_messages(self, task_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE task_id = ? ORDER BY timestamp ASC", (task_id,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "from_agent": r["from_agent"],
                "to_agent": r["to_agent"],
                "type": r["type"],
                "payload": json.loads(r["payload_json"]),
                "parent_id": r["parent_id"],
                "timestamp": r["timestamp"],
                "priority": r["priority"],
            }
            for r in rows
        ]

    # ── Context (key-value metadata) ─────────────────────────────────────

    def save_context(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO context (key, value, metadata_json) VALUES (?, ?, ?)",
            (key, value, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_context(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM context WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return {"key": row["key"], "value": row["value"], "metadata": json.loads(row["metadata_json"])}

    def search_context(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Simple keyword search over stored contexts (fallback when no vector store)."""
        rows = self._conn.execute(
            "SELECT * FROM context WHERE value LIKE ? LIMIT ?",
            (f"%{query}%", top_k),
        ).fetchall()
        return [
            {"key": r["key"], "value": r["value"], "metadata": json.loads(r["metadata_json"])}
            for r in rows
        ]
