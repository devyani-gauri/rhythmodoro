# rhythmodoro_code

Packaged core + app entry points for Rhythmodoro. Older files remain unchanged; this folder is the consolidated package you can import or run.

## Install deps

```bash
pip install -r requirements.txt
```

## Use as a library

```python
from rhythmodoro_code import get_collection, run_playlist
coll = get_collection()
res = run_playlist(coll, task_description="45-minute commute", target_vibe="Upbeat")
```

## Run Streamlit UI

```bash
streamlit run src/rhythmodoro_code/app_streamlit.py
```

## Run FastAPI server

```bash
uvicorn src.rhythmodoro_code.app_api:app --reload
```

Notes:
- Expects Milvus reachable at MILVUS_HOST:PORT and collection name MILVUS_COLLECTION (defaults: localhost:19530, embedded_music_data).
- If Ollama isn’t running, time estimation falls back gracefully.
