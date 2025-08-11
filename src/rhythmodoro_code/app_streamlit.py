import math
import os
import sys

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from rhythmodoro_code.core import get_collection, run_playlist, format_ms, normalize_artists, estimate_song_count
import streamlit as st

st.set_page_config(page_title="Rhythmodoro", page_icon="🎵", layout="centered")

st.title("Rhythmodoro 🎵")

with st.sidebar:
    st.header("Connection")
    host = st.text_input("Milvus Host", os.getenv("MILVUS_HOST", "localhost"))
    port = st.number_input("Port", value=int(os.getenv("MILVUS_PORT", "19530")))
    collection_name = st.text_input("Collection", os.getenv("MILVUS_COLLECTION", "embedded_music_data"))
    if st.button("Connect", type="primary"):
        st.session_state["_reload"] = True

if "_coll" not in st.session_state or st.session_state.get("_reload"):
    try:
        st.session_state["_coll"] = get_collection(host, port, collection_name)
        st.success("Connected to Milvus and loaded collection.")
    except Exception as e:
        st.error(f"Failed to connect/load collection: {e}")
    st.session_state["_reload"] = False

coll = st.session_state.get("_coll")

st.subheader("Your Task and Vibe")
col1, col2 = st.columns([3, 2])
with col1:
    task_desc = st.text_input("Describe your task", "45-minute commute with upbeat energy")
with col2:
    vibes = ["", "Chill", "Pop", "Dance", "Acoustic", "Upbeat", "Groove", "Lofi", "Soft Haze", "Pump Up", "Midnight Blues"]
    default_vibe = st.session_state.get("_target_vibe", "Upbeat")
    try:
        idx = vibes.index(default_vibe)
    except ValueError:
        idx = 0
    target_vibe = st.selectbox(
        "Pick a vibe",
        vibes,
        index=idx,
    )
    st.session_state["_target_vibe"] = target_vibe or st.session_state.get("_target_vibe")

# Quick-pick common vibes
q1, q2, q3, q4, q5 = st.columns(5)
if q1.button("Chill"):
    st.session_state["_target_vibe"] = "Chill"
if q2.button("Upbeat"):
    st.session_state["_target_vibe"] = "Upbeat"
if q3.button("Groove"):
    st.session_state["_target_vibe"] = "Groove"
if q4.button("Lofi"):
    st.session_state["_target_vibe"] = "Lofi"
if q5.button("Pop"):
    st.session_state["_target_vibe"] = "Pop"

with st.expander("Advanced options", expanded=False):
    include_blended = st.checkbox("Include 'Blended'", value=True)
    avoid_adjacent = st.checkbox("Avoid adjacent same artist", value=True)
    time_budget_mode = st.checkbox("Time-accurate capping", value=True)
    blended_max_ratio = st.slider("Max blended ratio", 0.0, 1.0, 0.4, 0.05)
    max_playlist_songs = st.slider("Max playlist songs", 1, 100, 50, 1)
if 'include_blended' not in locals():
    # Defaults if expander not opened yet
    include_blended = True
    avoid_adjacent = True
    time_budget_mode = True
    blended_max_ratio = 0.4
    max_playlist_songs = 50

colA, colB = st.columns(2)
with colA:
    do_estimate = st.button("Estimate songs", type="secondary")
with colB:
    do_generate = st.button("Generate playlist", type="primary")

if do_estimate:
    if not coll:
        st.error("Not connected to Milvus.")
    else:
        with st.spinner("Estimating…"):
            est = estimate_song_count(
                collection=coll,
                task_description=task_desc,
                max_playlist_songs=max_playlist_songs,
            )
        st.session_state["_estimate"] = est

est = st.session_state.get("_estimate")
if est:
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Estimated songs", int(est.get('recommended_count') or 0))
    with m2:
        st.metric("Estimated minutes", int(est.get('minutes_estimate') or 0))
    desired_song_count = st.number_input(
        "Override number of songs (optional)", min_value=0, max_value=100, value=int(est.get("recommended_count") or 0), step=1,
        help="0 means use estimate"
    )
else:
    desired_song_count = 0

if do_generate:
    if not coll:
        st.error("Not connected to Milvus.")
    else:
        with st.spinner("Searching..."):
            res = run_playlist(
                collection=coll,
                task_description=task_desc,
                target_vibe=target_vibe or None,
                include_blended=include_blended,
                time_budget_mode=time_budget_mode,
                blended_max_ratio=blended_max_ratio,
                max_playlist_songs=max_playlist_songs,
                avoid_adjacent_same_artist=avoid_adjacent,
                desired_song_count=(int(desired_song_count) if int(desired_song_count) > 0 else None),
            )
        selected = res.get("selected_count", 0)
        total_ms = int(res.get("total_ms") or 0)
        mins = math.floor(total_ms / 60000)
        st.success(f"This task should take {selected} songs to complete!")
        st.caption(f"Estimated task duration: {res.get('minutes_estimate') or 0} minutes")
        eff = res.get("effective_target_count")
        rec = res.get("recommended_count")
        st.write(f"Playlist suggestions: {selected} songs, total duration {mins} min ({format_ms(total_ms)})")
        if eff and rec and eff != rec:
            st.caption(f"Overridden target songs: {eff} (estimate was {rec})")

        songs = res.get("songs", [])
        for i, s in enumerate(songs, 1):
            art = ", ".join(normalize_artists(s.get("artists")))
            st.write(f"{i}. {s.get('name')} — {art} | {s.get('vibe')} | {format_ms(s.get('duration_ms') or 0)}")
