from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, sys

# Ensure package root (../) is on sys.path, then import via absolute package path
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from rhythmodoro_code.core import get_collection, run_playlist

app = FastAPI(title="Rhythmodoro API (packaged)")

class PlaylistRequest(BaseModel):
    task_description: Optional[str] = None
    target_vibe: Optional[str] = None
    include_blended: bool = True
    time_budget_mode: bool = True
    blended_max_ratio: float = 0.4
    max_playlist_songs: int = 50
    avoid_adjacent_same_artist: bool = True
    desired_song_count: Optional[int] = None

@app.on_event("startup")
async def _startup():
    global _coll
    _coll = get_collection()

@app.post("/playlist")
async def playlist(req: PlaylistRequest):
    try:
        res = run_playlist(
            collection=_coll,
            task_description=req.task_description,
            target_vibe=req.target_vibe,
            include_blended=req.include_blended,
            time_budget_mode=req.time_budget_mode,
            blended_max_ratio=req.blended_max_ratio,
            max_playlist_songs=req.max_playlist_songs,
            avoid_adjacent_same_artist=req.avoid_adjacent_same_artist,
            desired_song_count=req.desired_song_count,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
