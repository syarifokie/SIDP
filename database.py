import sqlite3
import os
from datetime import datetime

_db_path = "data/sidp.db"

def init_db(path="data/sidp.db"):
    global _db_path
    _db_path = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                use_case  TEXT,
                status    TEXT,
                missing   TEXT,
                detected  TEXT
            )
        """)
        c.commit()

def _conn():
    conn = sqlite3.connect(_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def insert_event(use_case, status, missing, detected):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        c.execute(
            "INSERT INTO events (timestamp,use_case,status,missing,detected) VALUES (?,?,?,?,?)",
            (ts, use_case, status, ", ".join(missing), ", ".join(detected))
        )
        c.commit()

def get_recent(limit=50, use_cases=None):
    with _conn() as c:
        if use_cases:
            placeholders = ",".join("?" * len(use_cases))
            rows = c.execute(
                f"SELECT * FROM events WHERE use_case IN ({placeholders}) ORDER BY id DESC LIMIT ?",
                (*use_cases, limit)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]

def get_summary(use_cases=None):
    with _conn() as c:
        if use_cases:
            placeholders = ",".join("?" * len(use_cases))
            row = c.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(status='COMPLIANT') as compliant,
                    SUM(status='WARNING')   as warning,
                    SUM(status='ALERT')     as alert
                FROM events WHERE use_case IN ({placeholders})
            """, use_cases).fetchone()
        else:
            row = c.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(status='COMPLIANT') as compliant,
                    SUM(status='WARNING')   as warning,
                    SUM(status='ALERT')     as alert
                FROM events
            """).fetchone()
    return dict(row)

def clear_events(use_cases=None):
    with _conn() as c:
        if use_cases:
            placeholders = ",".join("?" * len(use_cases))
            c.execute(f"DELETE FROM events WHERE use_case IN ({placeholders})", use_cases)
        else:
            c.execute("DELETE FROM events")
        c.commit()