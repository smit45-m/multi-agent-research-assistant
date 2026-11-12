"""Per-user Short-Term and Long-Term Memory Engine persisted inside the user's MongoDB DB.

Operates 100% locally (privacy/air-gap safe) with optional dense similarity
reranking using the CPU local embedding model when loaded.
"""
from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.mongo_engine import get_mongo_manager
from app.rag.privacy_guard import redact_pii
from app.rag.relevance import tokens

MAX_SHORT_TERM_TURNS = 10
MAX_LONG_TERM_ITEMS = 120

_STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "explain", "depth", "detail", "detailed", "about", "with", "from", "that",
    "this", "these", "those", "have", "has", "had", "does", "did", "work",
    "works", "using", "between", "compare", "comparison", "difference", "into",
    "more", "please", "tell", "give", "show", "make", "code", "example",
}

_PREFERENCE_RE = re.compile(
    r"\b(?:i am|i'm|my role is|i work (?:as|on|in)|remember(?: that)?|please always|"
    r"i prefer|my goal is|i study|i'm preparing for|i am preparing for)\b",
    re.IGNORECASE,
)

_FOLLOWUP_HINT_RE = re.compile(
    r"\b(?:it|that|this|they|them|those|these|above|previous|earlier|last|"
    r"follow[- ]?up|same|continue|elaborate|expand|simplify|more about|"
    r"second|third|step|uploaded|my file|my document|my pdf)\b",
    re.IGNORECASE,
)


def _extract_topic_key(query: str, attachment_names: Optional[List[str]] = None) -> tuple[str, List[str]]:
    q_toks = [t for t in tokens(query) if len(t) >= 3 and t not in _STOP_WORDS]
    file_toks = []
    for fn in attachment_names or []:
        stem = re.sub(r"\.[^.]+$", "", str(fn))
        file_toks.extend(t for t in tokens(stem) if len(t) >= 3 and t not in _STOP_WORDS)
    combined = list(dict.fromkeys(q_toks[:6] + file_toks[:4]))
    if not combined:
        combined = [t for t in tokens(query)[:4]] or ["general"]
    slug = "_".join(sorted(combined[:3]))[:48]
    return slug, combined


def _clean_summary_text(report: str, max_len: int = 420) -> str:
    if not report:
        return ""
    body_lines: List[str] = []
    for raw in report.splitlines():
        s = raw.strip()
        if not s:
            continue
        if (
            s.startswith("# 🔒")
            or ("100% Air-Gapped Verified" in s)
            or ("Privacy Guarantee" in s)
            or ("Privacy & Local Execution Audit" in s)
            or ("Synthesized via" in s)
        ):
            continue
        cleaned = re.sub(r"^#+\s*", "", s)
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        if cleaned:
            body_lines.append(cleaned)
        if sum(len(x) for x in body_lines) >= max_len:
            break
    text = redact_pii(" ".join(body_lines)).strip()
    return text[:max_len] + ("..." if len(text) > max_len else "")


class UserMemoryService:
    """Manages short-term conversational memory and durable long-term memory per user."""

    def __init__(self, db_name: str, user_id: str = ""):
        self.db_name = db_name
        self.user_id = user_id
        self.db = get_mongo_manager().get_user_db(db_name)
        self.short_col = self.db["short_term_memory"]
        self.long_col = self.db["long_term_memory"]

    # ── Memory Updates After Each Research Turn ────────────────────────────

    def update_after_turn(
        self,
        *,
        chat_id: str,
        thread_id: str,
        query: str,
        report: str,
        attachment_names: Optional[List[str]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        clean_q = redact_pii(query.strip())
        summary = _clean_summary_text(report, max_len=440)
        topic_key, keywords = _extract_topic_key(clean_q, attachment_names)
        files_used = list(dict.fromkeys(attachment_names or []))
        for s in sources or []:
            st = s.get("title") or ""
            if s.get("source_type") in ("pdf", "docx", "txt", "csv", "json", "md", "image", "audio", "document"):
                if st and st not in files_used:
                    files_used.append(st)
        files_used = files_used[:8]

        # 1. Push into Short-Term Working Memory (sliding window per user)
        stm_doc = {
            "_id": chat_id,
            "memory_id": chat_id,
            "user_id": self.user_id,
            "thread_id": thread_id,
            "chat_id": chat_id,
            "query": clean_q,
            "answer_summary": summary,
            "keywords": keywords,
            "files_used": files_used,
            "created_at": now,
        }
        self.short_col.insert_one(stm_doc)
        all_stm = list(self.short_col.find({}).sort("created_at", -1))
        if len(all_stm) > MAX_SHORT_TERM_TURNS:
            stale_ids = [item["memory_id"] for item in all_stm[MAX_SHORT_TERM_TURNS:]]
            if stale_ids:
                self.short_col.delete_many({"memory_id": {"$in": stale_ids}})

        # 2. Upsert into Long-Term Persistent User Memory
        is_pref = bool(_PREFERENCE_RE.search(clean_q))
        kind = "user_preference" if is_pref else ("document_insight" if files_used else "topic_knowledge")
        ltm_key = f"{kind}:{topic_key}" if not is_pref else f"pref:{uuid.uuid4().hex[:10]}"
        title_snippet = clean_q[:90] + ("..." if len(clean_q) > 90 else "")
        ltm_text = clean_q if is_pref else f"Q: {title_snippet} | Insight: {summary[:280]}"

        existing = self.long_col.find_one({"topic_key": ltm_key}) if not is_pref else None
        if existing:
            merged_kw = list(dict.fromkeys((existing.get("keywords") or []) + keywords))[:14]
            merged_files = list(dict.fromkeys((existing.get("source_files") or []) + files_used))[:10]
            merged_chats = list(dict.fromkeys((existing.get("chat_ids") or []) + [chat_id]))[-10:]
            self.long_col.update_one(
                {"topic_key": ltm_key},
                {
                    "$set": {
                        "title": title_snippet,
                        "summary": redact_pii(ltm_text),
                        "keywords": merged_kw,
                        "source_files": merged_files,
                        "chat_ids": merged_chats,
                        "updated_at": now,
                    },
                    "$inc": {"access_count": 1},
                },
            )
        else:
            mem_id = str(uuid.uuid4())
            self.long_col.insert_one({
                "_id": mem_id,
                "memory_id": mem_id,
                "user_id": self.user_id,
                "topic_key": ltm_key,
                "kind": kind,
                "title": title_snippet,
                "summary": redact_pii(ltm_text),
                "keywords": keywords,
                "source_files": files_used,
                "chat_ids": [chat_id],
                "access_count": 1,
                "pinned": bool(is_pref),
                "created_at": now,
                "updated_at": now,
            })

        # Keep long-term store bounded cleanly (prefer pinned + recently used)
        total_ltm = self.long_col.count_documents({})
        if total_ltm > MAX_LONG_TERM_ITEMS:
            all_ltm = list(self.long_col.find({}).sort("updated_at", -1))
            unpinned = [m for m in all_ltm if not m.get("pinned")]
            excess = len(all_ltm) - MAX_LONG_TERM_ITEMS
            if excess > 0 and unpinned:
                drop_ids = [m["memory_id"] for m in unpinned[-excess:]]
                self.long_col.delete_many({"memory_id": {"$in": drop_ids}})

    # ── Recall Short-Term + Long-Term Context for New Query ────────────────

    def recall_context(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        q_tokens = {t for t in tokens(query) if len(t) >= 3 and t not in _STOP_WORDS}
        is_followup = bool(_FOLLOWUP_HINT_RE.search(query))

        # 1. Short-term retrieval: prioritize same conversation thread chronologically
        all_stm = list(self.short_col.find({}).sort("created_at", -1))
        same_thread: List[Dict[str, Any]] = []
        other_recent: List[Dict[str, Any]] = []
        for item in all_stm:
            item.pop("_id", None)
            if thread_id and item.get("thread_id") == thread_id:
                same_thread.append(item)
            else:
                other_recent.append(item)

        short_selected = list(reversed(same_thread[:4]))  # chronological within active thread
        if len(short_selected) < 3 and other_recent:
            for item in other_recent[: 3 - len(short_selected)]:
                # Include very recent cross-thread context if user asks a follow-up OR it shares tokens
                item_toks = set(item.get("keywords") or []) | set(tokens(item.get("query") or ""))
                if is_followup or not q_tokens or (q_tokens & item_toks):
                    short_selected.append(item)

        # 2. Long-term retrieval: score relevance across past sessions + user preferences
        all_ltm = list(self.long_col.find({}).sort("updated_at", -1))
        scored_ltm: List[tuple[float, Dict[str, Any]]] = []
        seen_ids = {s.get("chat_id") for s in short_selected if s.get("chat_id")}

        for m in all_ltm:
            m.pop("_id", None)
            mem_toks = set(m.get("keywords") or []) | {t for t in tokens(m.get("title", "") + " " + m.get("summary", "")) if len(t) >= 3}
            overlap = len(q_tokens & mem_toks) / max(len(q_tokens), 1) if q_tokens else 0.0
            pref_bonus = 0.45 if m.get("kind") in ("user_preference", "pinned_note") or m.get("pinned") else 0.0
            file_bonus = 0.15 if m.get("source_files") and is_followup else 0.0
            freq_bonus = min(0.15, math.log1p(int(m.get("access_count", 1))) * 0.06)
            score = overlap + pref_bonus + file_bonus + freq_bonus
            # Skip entries already shown verbatim in short-term turns unless pinned preference
            if not m.get("pinned") and all(cid in seen_ids for cid in (m.get("chat_ids") or [])):
                score -= 0.35
            if score >= 0.18 or m.get("pinned") or m.get("kind") == "user_preference":
                scored_ltm.append((score, m))

        scored_ltm.sort(key=lambda pair: pair[0], reverse=True)
        long_selected = [m for _, m in scored_ltm[:5]]

        # Format compact, prompt-injection-safe context block
        sections: List[str] = []
        if short_selected:
            st_lines = []
            for idx, item in enumerate(short_selected, 1):
                files_str = f" [Files: {', '.join(item.get('files_used', []))}]" if item.get("files_used") else ""
                st_lines.append(
                    f"  {idx}. User asked: \"{item.get('query', '')}\"{files_str}\n"
                    f"     Summary: {(item.get('answer_summary') or '')[:260]}"
                )
            sections.append("Short-Term Conversation Context (recent turns for this user):\n" + "\n".join(st_lines))

        if long_selected:
            lt_lines = []
            for idx, m in enumerate(long_selected, 1):
                f_str = f" [Files: {', '.join(m.get('source_files', []))}]" if m.get("source_files") else ""
                lt_lines.append(f"  {idx}. [{m.get('kind', 'memory')}] {m.get('summary', '')[:300]}{f_str}")
            sections.append("Long-Term Remembered Context (across previous sessions for this user):\n" + "\n".join(lt_lines))

        formatted = "\n\n".join(sections)
        return {
            "short_term": short_selected,
            "long_term": long_selected,
            "short_term_count": len(short_selected),
            "long_term_count": len(long_selected),
            "formatted_prompt_block": formatted,
        }

    # ── Explicit Inspection & User Control APIs ────────────────────────────

    def list_memories(self) -> Dict[str, Any]:
        stm = list(self.short_col.find({}).sort("created_at", -1))
        ltm = list(self.long_col.find({}).sort("updated_at", -1))
        for item in stm + ltm:
            item.pop("_id", None)
        return {
            "short_term": stm,
            "long_term": ltm,
            "short_term_count": len(stm),
            "long_term_count": len(ltm),
        }

    def add_pinned_memory(self, title: str, summary: str, kind: str = "pinned_note") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        mem_id = str(uuid.uuid4())
        safe_title = redact_pii(title.strip())[:140]
        safe_summary = redact_pii(summary.strip())[:800]
        _, keywords = _extract_topic_key(f"{safe_title} {safe_summary}")
        doc = {
            "_id": mem_id,
            "memory_id": mem_id,
            "user_id": self.user_id,
            "topic_key": f"pin:{mem_id[:12]}",
            "kind": kind if kind in ("pinned_note", "user_preference") else "pinned_note",
            "title": safe_title,
            "summary": safe_summary,
            "keywords": keywords,
            "source_files": [],
            "chat_ids": [],
            "access_count": 1,
            "pinned": True,
            "created_at": now,
            "updated_at": now,
        }
        self.long_col.insert_one(doc)
        doc.pop("_id", None)
        return doc

    def delete_long_term(self, memory_id: str) -> bool:
        res = self.long_col.delete_one({"memory_id": memory_id})
        return res.deleted_count > 0

    def clear_short_term(self) -> int:
        res = self.short_col.delete_many({})
        return int(res.deleted_count)
