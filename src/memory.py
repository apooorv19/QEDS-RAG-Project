import sqlite3
from pathlib import Path

# DB will live at project root (/app/chat_memory.db in Docker)
DB_PATH = Path(__file__).resolve().parent.parent / "chat_memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_memory (
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_message(user_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_memory (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()


def load_messages(user_id: str, limit: int = 8):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT role, content
        FROM chat_memory
        WHERE user_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    """, (user_id, limit))
    rows = c.fetchall()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_user_memory(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "DELETE FROM chat_memory WHERE user_id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()
