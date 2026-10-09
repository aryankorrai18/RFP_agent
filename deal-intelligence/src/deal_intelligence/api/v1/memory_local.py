"""The free local memory backend: the same two banks as Hindsight, kept in a SQLite FTS5 file.

`LocalMemory` implements the interactions bank (memory.Memory) and `LocalLessons` the lessons bank
(lessons.LessonsMemory), so the app runs with no Hindsight account. Recall is keyword search ranked by
bm25, not semantic search; the structural gate and the SQLite re-check in retrieval.py still decide
what may be used. Reflect and the playbook are computed from the stored lessons with no model.

Several banks can share one file (each row carries its bank), and a workspace's file lives in the
workspace's own folder, so two workspaces never see each other's documents. Connections are short-lived
(one per call, behind a lock), so the file is never held open and a workspace folder can be deleted.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .lessons import LessonBrief, LessonHit
from .memory import MemoryItem, MemoryUnavailable, RecallHit

SEP = "\x1f"  # tags are stored as \x1f-joined text; instr() on "\x1ftag\x1f" is an exact tag match
MAX_QUERY_TOKENS = 48
LOCAL_FOOTER = "Computed locally from recorded outcomes, not by Hindsight."
REFLECT_FOOTER = "This was summarised locally without a model."
LESSON_TYPE = "world"

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its just me more most my no nor not of off on once only or other our out over
own same she should so some such than that the their theirs them then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
""".split())

_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
    "bank UNINDEXED, code UNINDEXED, tags UNINDEXED, content, tokenize='porter unicode61')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5("
    "bank UNINDEXED, code UNINDEXED, tags UNINDEXED, content, tokenize='porter unicode61')",
)
_lock = threading.Lock()


def fts_query(text: str) -> str:
    """Natural language to an FTS5 query: lowercase word tokens, stop words and one-letter tokens
    dropped, each quoted (so no token can act as an operator) and joined with OR. '' means no query."""
    seen: dict[str, None] = {}
    for token in _TOKEN.findall((text or "").lower()):
        if len(token) > 1 and token not in _STOP:
            seen.setdefault(token)
        if len(seen) >= MAX_QUERY_TOKENS:
            break
    return " OR ".join(f'"{token}"' for token in seen)


def _pack(tags: list[str]) -> str:
    return SEP + SEP.join(tags) + SEP if tags else ""


def _unpack(packed: str) -> list[str]:
    return [t for t in packed.split(SEP) if t]


def _tag_clause(tags: list[str], joiner: str) -> tuple[str, list[str]]:
    if not tags:
        return "", []
    return " AND (" + f" {joiner} ".join("instr(tags, ?) > 0" for _ in tags) + ")", [f"{SEP}{t}{SEP}" for t in tags]


class _Store:
    def __init__(self, path: Path | str, bank: str):
        self.path = Path(path)
        self.bank = bank

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for statement in _SCHEMA:
                conn.execute(statement)
            return conn
        except sqlite3.Error as exc:
            raise MemoryUnavailable(f"Local memory file {self.path} is not usable: {exc}") from exc

    def _run(self, work):  # noqa: ANN001, ANN202
        with _lock:
            try:
                with closing(self._connect()) as conn:
                    result = work(conn)
                    conn.commit()
                    return result
            except sqlite3.Error as exc:
                raise MemoryUnavailable(f"Local memory failed: {exc}") from exc

    def _upsert(self, conn: sqlite3.Connection, table: str, code: str, tags: list[str], content: str) -> None:
        conn.execute(f"DELETE FROM {table} WHERE bank = ? AND code = ?", (self.bank, code))
        conn.execute(f"INSERT INTO {table} (bank, code, tags, content) VALUES (?, ?, ?, ?)",
                     (self.bank, code, _pack(tags), content))

    def _search(self, conn: sqlite3.Connection, table: str, query: str, tags: list[str], joiner: str,
                limit: int) -> list[tuple[str, str, str, float]]:
        match = fts_query(query)
        if not match or limit <= 0:
            return []
        clause, params = _tag_clause(tags, joiner)
        rows = conn.execute(
            f"SELECT code, tags, content, bm25({table}) AS score FROM {table} "
            f"WHERE {table} MATCH ? AND bank = ?{clause} ORDER BY score, code LIMIT ?",
            (match, self.bank, *params, limit),
        ).fetchall()
        return [(code, packed, content, -float(score)) for code, packed, content, score in rows]  # higher is better


class LocalMemory(_Store):
    """Interactions bank. recall needs every requested tag on a document, like Hindsight's all_strict."""

    async def ensure_bank(self) -> None:
        self._run(lambda conn: None)

    async def retain(self, item: MemoryItem) -> None:
        self._run(lambda conn: self._upsert(conn, "memory_fts", item.code, list(item.tags), item.content))

    async def delete(self, code: str) -> None:
        self._run(lambda conn: conn.execute("DELETE FROM memory_fts WHERE bank = ? AND code = ?", (self.bank, code)))

    async def recall(self, query: str, tags: list[str], limit: int) -> list[RecallHit]:
        rows = self._run(lambda conn: self._search(conn, "memory_fts", query, list(tags), "AND", limit))
        return [RecallHit(code=code, rank=n, final=score) for n, (code, _tags, _content, score) in enumerate(rows, start=1)]

    async def healthy(self) -> bool:
        return True

    async def extraction_mode(self) -> str | None:
        return "local"

    async def close(self) -> None:
        return None


# ---- lessons ---------------------------------------------------------------------------------------


def _signal(tags: list[str]) -> str:
    return next((t.removeprefix("signal:") for t in tags if t.startswith("signal:")), "neutral")


def _first_sentence(text: str, limit: int = 260) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


def _tally(rows: list[tuple[str, list[str], str]]) -> tuple[int, int, int]:
    counts = Counter(_signal(tags) for _code, tags, _text in rows)
    return counts["positive"], counts["negative"], counts["neutral"]


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


class LocalLessons(_Store):
    """Lessons bank. recall matches ANY requested tag (like tags_match="any"); reflect and the playbook
    are built deterministically from the stored lessons."""

    async def ensure_bank(self) -> None:
        self._run(lambda conn: None)

    async def retain(self, items: list[dict]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for item in items:
                self._upsert(conn, "lessons_fts", item["document_id"], list(item.get("tags") or []), item["content"])

        self._run(work)

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]:
        rows = self._run(lambda conn: self._search(conn, "lessons_fts", query, list(tags), "OR", limit))
        return [LessonHit(text=content, tags=_unpack(packed), document_id=code, type=LESSON_TYPE, rank=n)
                for n, (code, packed, content, _score) in enumerate(rows, start=1)]

    def _matching(self, conn: sqlite3.Connection, query: str, tags: list[str]) -> list[tuple[str, list[str], str]]:
        """Every lesson carrying any of the tags (all when none), most relevant to the query first."""
        ranked = self._search(conn, "lessons_fts", query, tags, "OR", 1_000_000)
        seen = {code for code, *_ in ranked}
        clause, params = _tag_clause(tags, "OR")
        rest = conn.execute(f"SELECT code, tags, content FROM lessons_fts WHERE bank = ?{clause} ORDER BY code",
                            (self.bank, *params)).fetchall()
        ordered = [(code, _unpack(packed), content) for code, packed, content, _score in ranked]
        return ordered + [(code, _unpack(packed), content) for code, packed, content in rest if code not in seen]

    async def reflect(self, query: str, tags: list[str] | None = None) -> LessonBrief:
        lessons = self._run(lambda conn: self._matching(conn, query, list(tags or [])))
        outcomes = [row for row in lessons if "kind:deal_outcome" in row[1]] or lessons
        if not outcomes:
            return LessonBrief(text=f"No recorded outcomes match this situation yet. {REFLECT_FOOTER}", based_on=[])
        won, lost_fixable, neutral = _tally(outcomes)
        parts = [f"{_plural(len(outcomes), 'recorded outcome')} match this situation: {won} won, "
                 f"{lost_fixable} lost for a reason a play could have changed, and {neutral} with no verdict on the plays."]
        top = lessons[:3]
        parts.append("Most relevant lessons: " + " ".join(f"({n}) {_first_sentence(text)}" for n, (_c, _t, text) in enumerate(top, start=1)))
        parts.append(REFLECT_FOOTER)
        return LessonBrief(
            text=" ".join(parts),
            based_on=[{"id": code, "text": text, "type": LESSON_TYPE} for code, _tags, text in top],
        )

    async def playbook(self, refresh: bool = False) -> dict | None:
        lessons = self._run(lambda conn: self._matching(conn, "", []))
        return {"content": _playbook_markdown(lessons), "last_refreshed_at": datetime.now(UTC).isoformat(),
                "is_stale": False}

    async def close(self) -> None:
        return None


_PLAY_NAME = re.compile(r"^Play (\S+?)(?: \(([^)]*)\))? was used in deal")
FAMILIES = (("objection", "Objections"), ("competitor", "Competitors"), ("industry", "Industries"),
            ("segment", "Segments"), ("loss", "Loss reasons"))


def _line(label: str, rows: list[tuple[str, list[str], str]]) -> str:
    won, lost, neutral = _tally(rows)
    return f"- {label}: {won} won, {lost} lost (fixable), {neutral} no verdict ({_plural(len(rows), 'deal')})"


def _playbook_markdown(lessons: list[tuple[str, list[str], str]]) -> str:
    outcomes = [row for row in lessons if "kind:deal_outcome" in row[1]]
    play_rows = [row for row in lessons if "kind:play_result" in row[1]]
    lines = ["# Deal playbook"]
    if not outcomes and not play_rows:
        lines += ["", "No closed deals have been recorded yet, so there is nothing to count."]
    else:
        won, lost, neutral = _tally(outcomes)
        lines += ["", f"Counts over {_plural(len(outcomes), 'closed deal')}: {won} won, {lost} lost for a reason a play "
                      f"could have changed, {neutral} with no verdict on the plays (for example lost on price)."]
        by_play: dict[str, list[tuple[str, list[str], str]]] = {}
        names: dict[str, str] = {}
        for row in play_rows:
            for tag in row[1]:
                if tag.startswith("play:"):
                    by_play.setdefault(tag.removeprefix("play:"), []).append(row)
                    if (match := _PLAY_NAME.match(row[2])) and match.group(2):
                        names[tag.removeprefix("play:")] = match.group(2)
        if by_play:
            lines += ["", "## Plays (each deal where the play was used)"]
            lines += [_line(f"{code} ({names[code]})" if code in names else code, rows)
                      for code, rows in sorted(by_play.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
        for prefix, title in FAMILIES:
            groups: dict[str, list[tuple[str, list[str], str]]] = {}
            for row in outcomes:
                for tag in row[1]:
                    if tag.startswith(prefix + ":"):
                        groups.setdefault(tag.removeprefix(prefix + ":"), []).append(row)
            if groups:
                lines += ["", f"## {title}"]
                lines += [_line(name.replace("_", " "), rows)
                          for name, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    lines += ["", "---", LOCAL_FOOTER]
    return "\n".join(lines)
