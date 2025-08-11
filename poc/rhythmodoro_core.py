"""
Core retrieval logic for Rhythmodoro: vibe-aware playlist generation with
- Milvus search
- Local LLM (Ollama) time estimation
- Time-accurate capping
- Blended cap
- Artist normalization and optional reordering to avoid adjacent duplicates

Usage:
    from rhythmodoro_core import get_collection, run_playlist
    coll = get_collection()
    result = run_playlist(coll, task_description="45-minute commute", target_vibe="Upbeat")
"""
from typing import List, Optional, Dict, Any, Tuple
import os
import json
import re
import requests

from pymilvus import connections, Collection

# Defaults / config via env
DEFAULT_MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
DEFAULT_MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "embedded_music_data")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

# Behavior toggles / limits
USE_TASK_TIME_DEFAULT = True
TIME_BUDGET_MODE_DEFAULT = True
BUDGET_OVERSHOOT_MS_DEFAULT = 15_000
MAX_PLAYLIST_SONGS_DEFAULT = 50
RESULT_LIMIT_DEFAULT = 10
BLENDED_MAX_RATIO_DEFAULT = 0.4
FETCH_MULTIPLIER_DEFAULT = 5
MAX_QUERY_WINDOW = 16384


def get_collection(
    host: str = DEFAULT_MILVUS_HOST,
    port: int = DEFAULT_MILVUS_PORT,
    name: str = DEFAULT_COLLECTION,
) -> Collection:
    connections.connect(alias="default", host=host, port=str(port))
    coll = Collection(name)
    coll.load()
    return coll


def ollama_up(base: str = OLLAMA_HOST) -> bool:
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL, expect_json: bool = False, num_predict: int = 64) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }
    if expect_json:
        payload["format"] = "json"
    r = requests.post(url, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def estimate_minutes_from_task(desc: str) -> int:
    if not desc or not desc.strip():
        return 0
    prompt = (
        "Estimate the total time required for this task in minutes. "
        "Return ONLY valid JSON like {\"minutes\": 45}.\nTask: " + desc.strip()
    )
    try:
        text = ollama_generate(prompt, expect_json=True, num_predict=64)
        try:
            obj = json.loads(text)
        except Exception:
            m = re.search(r"\{[^\}]*\}", text)
            obj = json.loads(m.group(0)) if m else {}
        minutes = int(float(obj.get("minutes", 0))) if isinstance(obj, dict) else 0
        return max(0, minutes)
    except Exception:
        return 0


def sample_avg_song_minutes(collection: Collection, sample_size: int = 5000) -> float:
    try:
        lim = min(sample_size, MAX_QUERY_WINDOW)
        rows = collection.query(expr="duration_ms >= 0", output_fields=["duration_ms"], limit=lim)
        if not rows:
            return 3.5
        vals = [r.get("duration_ms") for r in rows if r.get("duration_ms") is not None]
        if not vals:
            return 3.5
        avg_ms = sum(vals) / len(vals)
        return max(1e-6, avg_ms / 60000.0)
    except Exception:
        return 3.5


def format_ms(ms: int) -> str:
    try:
        total_sec = int(round(ms / 1000))
        m, s = divmod(total_sec, 60)
        return f"{m}:{s:02d}"
    except Exception:
        return "0:00"


def normalize_artists(a) -> List[str]:
    if a is None:
        return []
    if isinstance(a, list):
        return [str(x) for x in a]
    if isinstance(a, str):
        s = a.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if parts:
                return parts
        return [s]
    return [str(a)]


def _primary_artist(hit) -> str:
    arts = normalize_artists(hit.entity.get("artists"))
    return arts[0] if arts else ""


def reorder_no_adjacent_same_artist(hits: List) -> List:
    pending = list(hits)
    out: List = []
    while pending:
        if not out:
            out.append(pending.pop(0))
            continue
        last = _primary_artist(out[-1])
        idx = None
        for i, h in enumerate(pending):
            if _primary_artist(h) != last:
                idx = i
                break
        if idx is None:
            out.append(pending.pop(0))
        else:
            out.append(pending.pop(idx))
    return out


def count_vibes(collection: Collection, canonical_labels: Optional[List[str]] = None) -> Dict[str, int]:
    labels = canonical_labels or [
        "Chill", "Pop", "Dance", "Acoustic", "Upbeat", "Groove",
        "Lofi", "Soft Haze", "Pump Up", "Midnight Blues",
    ]
    counts: Dict[str, int] = {}
    for lab in labels + ["Blended"]:
        rows = collection.query(expr=f'vibe == "{lab}"', output_fields=["id"], limit=MAX_QUERY_WINDOW)
        counts[lab] = len(rows)
    return counts


def run_playlist(
    collection: Collection,
    task_description: Optional[str],
    target_vibe: Optional[str],
    *,
    include_blended: bool = True,
    time_budget_mode: bool = TIME_BUDGET_MODE_DEFAULT,
    blended_max_ratio: float = BLENDED_MAX_RATIO_DEFAULT,
    fetch_multiplier: int = FETCH_MULTIPLIER_DEFAULT,
    max_playlist_songs: int = MAX_PLAYLIST_SONGS_DEFAULT,
    budget_overshoot_ms: int = BUDGET_OVERSHOOT_MS_DEFAULT,
    default_count: int = RESULT_LIMIT_DEFAULT,
    seed_song_name: Optional[str] = None,
    avoid_adjacent_same_artist: bool = True,
    use_task_time: bool = USE_TASK_TIME_DEFAULT,
) -> Dict[str, Any]:
    # Time estimate
    minutes_estimate = 0
    avg_song_minutes = None
    if task_description and use_task_time and ollama_up():
        minutes_estimate = estimate_minutes_from_task(task_description) or 0
        if minutes_estimate > 0:
            avg_song_minutes = sample_avg_song_minutes(collection)

    # Recommended count
    if minutes_estimate > 0 and avg_song_minutes:
        recommended_count = int(max(1, min(max_playlist_songs, round(minutes_estimate / max(1e-6, avg_song_minutes)))))
    else:
        recommended_count = default_count

    # Build filter
    filter_expr = None
    if target_vibe and isinstance(target_vibe, str) and target_vibe.strip():
        filter_expr = f'vibe in ["{target_vibe}", "Blended"]' if include_blended else f'vibe == "{target_vibe}"'

    # Seed selection
    def get_one(expr: str):
        try:
            res = collection.query(expr=expr, output_fields=["embedding", "vibe", "name", "artists"], limit=1)
            return res[0] if res else None
        except Exception:
            return None

    seed_row = None
    if seed_song_name:
        seed_row = get_one(f'name == "{seed_song_name}"')
    else:
        if target_vibe:
            seed_row = get_one(f'vibe == "{target_vibe}"')
        if not seed_row:
            seed_row = get_one('vibe != "" and vibe != "Blended"') or get_one('vibe != ""')
    if not seed_row:
        raise ValueError("No seed candidate found. Check that the collection has 'vibe' populated.")

    query_embedding = seed_row["embedding"]

    # Search: ensure ef (search breadth) > k (fetch_k) for HNSW
    fetch_k = min(max(recommended_count * fetch_multiplier, recommended_count), MAX_QUERY_WINDOW)
    ef = max(64, fetch_k + 16)
    search_params = {"metric_type": "L2", "params": {"ef": ef}}
    raw_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=fetch_k,
        output_fields=["id", "name", "artists", "vibe", "duration_ms", "embedding_json"],
        expr=filter_expr,
    )[0]

    # Post-process
    max_blended = int(round(blended_max_ratio * recommended_count)) if include_blended else 0
    final: List = []
    blended_used = 0

    use_time_budget = time_budget_mode and (minutes_estimate or 0) > 0
    if use_time_budget:
        target_ms = int(minutes_estimate * 60_000)
        running_ms = 0
        for hit in raw_results:
            this_vibe = hit.entity.get("vibe") or ""
            is_blended = (this_vibe == "Blended")
            if is_blended and blended_used >= max_blended:
                continue
            dur = hit.entity.get("duration_ms")
            if not isinstance(dur, (int, float)) or dur <= 0:
                dur = int(3.5 * 60_000)
            if running_ms + dur > target_ms + budget_overshoot_ms:
                break
            final.append(hit)
            running_ms += dur
            if is_blended:
                blended_used += 1
            if len(final) >= max_playlist_songs:
                break
        if not final and len(raw_results) > 0:
            top = raw_results[0]
            top_dur = top.entity.get("duration_ms") or int(3.5 * 60_000)
            if top_dur <= target_ms + budget_overshoot_ms:
                final = [top]
    else:
        for hit in raw_results:
            this_vibe = hit.entity.get("vibe") or ""
            is_blended = (this_vibe == "Blended")
            if is_blended and blended_used >= max_blended:
                continue
            final.append(hit)
            if is_blended:
                blended_used += 1
            if len(final) >= recommended_count:
                break

    if avoid_adjacent_same_artist and len(final) > 1:
        final = reorder_no_adjacent_same_artist(final)

    total_ms = sum((h.entity.get("duration_ms") or int(3.5 * 60_000)) for h in final)
    selected_count = len(final)

    return {
        "songs": [
            {
                "name": h.entity.get("name"),
                "artists": normalize_artists(h.entity.get("artists")),
                "vibe": h.entity.get("vibe"),
                "duration_ms": h.entity.get("duration_ms") or int(3.5 * 60_000),
                "score": h.score,
            }
            for h in final
        ],
        "total_ms": total_ms,
        "selected_count": selected_count,
        "recommended_count": recommended_count,
        "minutes_estimate": minutes_estimate,
        "avg_song_minutes": avg_song_minutes,
        "target_vibe": target_vibe,
        "include_blended": include_blended,
        "avoid_adjacent_same_artist": avoid_adjacent_same_artist,
    }
