"""
Demo Videos Page
"""

import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from utils.api_client import CrossIDClient
from utils.video_utils import get_video_info


# -----------------------------
# Helper function
# -----------------------------
def get_browser_compatible_video(video_path: str) -> str:
    """
    Convert video to H.264 if needed for browser playback.
    Returns path to browser-compatible video.
    """
    video_path = Path(video_path)

    # Create cache directory for converted videos
    cache_dir = Path(tempfile.gettempdir()) / "crossid_preview_cache"
    cache_dir.mkdir(exist_ok=True)

    # Check if already converted
    cache_file = cache_dir / f"{video_path.stem}_h264.mp4"
    if cache_file.exists():
        return str(cache_file)

    try:
        # Convert with FFmpeg
        result = subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y',
            str(cache_file)
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and cache_file.exists():
            return str(cache_file)
        else:
            return str(video_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return str(video_path)

# -----------------------------
# Session state defaults
# -----------------------------
if "demo_processing" not in st.session_state:
    st.session_state.demo_processing = False
if "demo_selected_video" not in st.session_state:
    st.session_state.demo_selected_video = None

# Snapshots at click-time
if "demo_snapshot_enable_reid" not in st.session_state:
    st.session_state.demo_snapshot_enable_reid = True
if "demo_snapshot_show_trails" not in st.session_state:
    st.session_state.demo_snapshot_show_trails = True
if "demo_snapshot_confidence" not in st.session_state:
    st.session_state.demo_snapshot_confidence = 0.5

# Persist last result
if "demo_result_stats" not in st.session_state:
    st.session_state.demo_result_stats = None
if "demo_result_filename" not in st.session_state:
    st.session_state.demo_result_filename = None

# Browser-compatible sources
if "demo_raw_video_src" not in st.session_state:
    st.session_state.demo_raw_video_src = None
if "demo_processed_video_src" not in st.session_state:
    st.session_state.demo_processed_video_src = None
if "demo_processed_video_bytes" not in st.session_state:
    st.session_state.demo_processed_video_bytes = None

# Show preview flag
if "demo_show_preview" not in st.session_state:
    st.session_state.demo_show_preview = False

# -----------------------------
# Back to home button
# -----------------------------
home_page = st.Page("pages/home.py")
if st.button("← Back to Home"):
    st.switch_page(home_page)

st.markdown("---")

st.title("Demo Videos")
st.markdown("Pre-loaded scenarios showcasing different tracking challenges")
st.info("""
**Re-ID Model Performance:**
- Rank-1 Accuracy: **99.05%**
- mAP: **70.62%**
- Trained on Market-1501 dataset
""")

# -----------------------------
# Initialize client
# -----------------------------
if "api_client" not in st.session_state:
    st.session_state.api_client = CrossIDClient()
client = st.session_state.api_client

# -----------------------------
# Get demo videos
# -----------------------------
try:
    demo_videos = client.list_demo_videos()
except Exception as e:
    st.error(f"Failed to load demo videos: {e}")
    st.stop()

if not demo_videos:
    st.warning("No demo videos available")
    st.stop()

# -----------------------------
# Video metadata
# -----------------------------
video_info = {
    "01_indoor_easy.mp4": {
        "title": "Indoor Easy",
        "description": "Indoor scene with 5-10 people, good lighting, minimal occlusions",
        "difficulty": "Easy",
    },
    "02_crowded_street.mp4": {
        "title": "Crowded Street",
        "description": "Busy street scene with 50+ pedestrians, occlusions, varying speeds",
        "difficulty": "Medium",
    },
    "03_sunset_backlit.mp4": {
        "title": "Sunset Backlit",
        "description": "Pedestrians backlit by sunset, silhouettes, challenging lighting",
        "difficulty": "Hard",
    },
    "04_extreme_challenge.mp4": {
        "title": "Extreme Challenge",
        "description": "Train station rush hour with 150+ people, extreme density",
        "difficulty": "Extreme",
    },
    "05_multicamera_lab.mp4": {
        "title": "Multi-Camera Re-ID - Laboratory",
        "description": "4 synchronized cameras with 4 people in a laboratory",
        "difficulty": "Medium",
    },
    "05_multicamera_basketball.mp4": {
        "title": "Multi-Camera Re-ID - Basketball Game",
        "description": "4 synchronized cameras for a basketball game with 10+ players",
        "difficulty": "Medium",
    },
}

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.markdown("### Processing Settings")
    enable_reid_checkbox = st.checkbox("Enable Re-ID", value=True, key="demo_enable_reid")
    show_trails_checkbox = st.checkbox("Show Trails", value=True, key="demo_show_trails")
    confidence_slider = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="demo_confidence")

# -----------------------------
# Grid always visible
# -----------------------------
st.markdown("---")
st.markdown("### Select Demo Video")

cols = st.columns(3)
for idx, video in enumerate(demo_videos):
    with cols[idx % 3]:
        filename = video["filename"]
        info = video_info.get(filename, {})

        st.markdown(f"**{info.get('title', filename)}**")
        st.caption(info.get("description", ""))
        st.caption(f"Difficulty: {info.get('difficulty', 'N/A')}")
        st.caption(f"Size: {video['size_mb']:.1f} MB")

        if st.button("Select", key=f"btn_{filename}", disabled=st.session_state.demo_processing):
            st.session_state.demo_selected_video = video
            st.session_state.demo_show_preview = True
            st.session_state.demo_processing = False

            # Clear previous processed
            st.session_state.demo_processed_video_src = None
            st.session_state.demo_processed_video_bytes = None
            st.session_state.demo_result_stats = None
            st.session_state.demo_result_filename = None

            # Set raw preview source
            video_path = video.get("path")
            if video_path:
                st.session_state.demo_raw_video_src = str(video_path)
            else:
                st.session_state.demo_raw_video_src = video.get("url")

            st.rerun()

# -----------------------------
# Preview section (if selected but not processing)
# -----------------------------
if st.session_state.demo_show_preview and not st.session_state.demo_processing:
    video = st.session_state.demo_selected_video
    if video:
        st.markdown("---")
        st.markdown("## Selected Video")

        filename = video["filename"]
        info = video_info.get(filename, {})

        st.markdown(f"### {info.get('title', filename)}")
        st.caption(info.get('description', ''))

        # Get video info
        try:
            video_metadata = get_video_info(video["path"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Resolution", f"{video_metadata['width']}x{video_metadata['height']}")
            col2.metric("FPS", f"{video_metadata['fps']:.1f}")
            col3.metric("Frames", video_metadata['frame_count'])
            col4.metric("Duration", f"{video_metadata['duration']:.1f}s")
        except Exception as e:
            st.warning(f"Could not read video info: {e}")

        # Show preview with conversion
        st.markdown("### Preview")
        if st.session_state.demo_raw_video_src:
            with st.spinner("Preparing browser-compatible preview..."):
                browser_compatible_src = get_browser_compatible_video(st.session_state.demo_raw_video_src)
            st.video(browser_compatible_src)
            st.caption("✓ Preview converted to H.264 for browser compatibility")

        # Process button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Process Video", type="primary", use_container_width=True):
                st.session_state.demo_snapshot_enable_reid = enable_reid_checkbox
                st.session_state.demo_snapshot_show_trails = show_trails_checkbox
                st.session_state.demo_snapshot_confidence = confidence_slider
                st.session_state.demo_processing = True
                st.session_state.demo_show_preview = False
                st.rerun()

        with col2:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.demo_selected_video = None
                st.session_state.demo_show_preview = False
                st.session_state.demo_raw_video_src = None
                st.rerun()

# -----------------------------
# Processing section
# -----------------------------
if st.session_state.demo_processing:
    st.markdown("---")
    st.markdown("## Processing")

    video = st.session_state.demo_selected_video
    if video is None:
        st.error("No video selected")
        st.session_state.demo_processing = False
        st.stop()

    reid_enabled = st.session_state.demo_snapshot_enable_reid
    trails_enabled = st.session_state.demo_snapshot_show_trails
    conf_threshold = st.session_state.demo_snapshot_confidence

    # 1) VIDEO SLOT (top)
    video_slot = st.empty()
    if st.session_state.demo_raw_video_src:
        # Show preview during processing
        with st.spinner("Loading preview..."):
            browser_compatible_src = get_browser_compatible_video(st.session_state.demo_raw_video_src)
        video_slot.video(browser_compatible_src)

    # 2) STATUS UI (under video)
    st.markdown("### Status")
    metric_ph = st.empty()
    progress_ph = st.empty()
    text_ph = st.empty()

    metric_ph.metric("Status", "Uploading")
    progress_bar = progress_ph.progress(0)
    text_ph.write("Starting...")

    try:
        video_path = video.get("path")
        if not video_path:
            st.error("Video path not found")
            st.session_state.demo_processing = False
            st.stop()

        # Upload
        text_ph.write("Uploading...")
        job_id = client.upload_video(
            video_path=str(video_path),
            enable_reid=reid_enabled,
            show_trails=trails_enabled,
            confidence_threshold=conf_threshold,
        )
        metric_ph.metric("Status", "Processing")
        progress_bar.progress(10)
        text_ph.write("Processing...")

        # Poll
        last_progress = -1
        final_status = None

        while True:
            status = client.get_job_status(job_id)
            final_status = status

            progress = int(status.get("progress", 0) * 100)
            if progress != last_progress:
                progress_bar.progress(min(max(progress, 0), 99))
                last_progress = progress

            text_ph.write(f"Processing: {progress}%")

            if status.get("status") == "completed":
                progress_bar.progress(100)
                metric_ph.metric("Status", "Complete")
                text_ph.write("Complete")
                break

            if status.get("status") == "failed":
                metric_ph.metric("Status", "Failed")
                st.error(f"Processing failed: {status.get('message')}")
                st.session_state.demo_processing = False
                st.stop()

            time.sleep(0.5)

        # Download result
        text_ph.write("Downloading result...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            processed_path = tmp.name
            client.download_result(job_id, processed_path)

        st.session_state.demo_processed_video_src = processed_path
        st.session_state.demo_result_stats = (final_status or {}).get("stats", {})
        st.session_state.demo_result_filename = video["filename"]

        with open(processed_path, "rb") as f:
            st.session_state.demo_processed_video_bytes = f.read()

        # Swap to processed video
        video_slot.video(st.session_state.demo_processed_video_src)

        st.session_state.demo_processing = False
        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.demo_processing = False

# -----------------------------
# Last result
# -----------------------------
if st.session_state.demo_processed_video_src and not st.session_state.demo_processing:
    st.markdown("---")
    st.markdown("## Results")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Processed Video")
        st.video(st.session_state.demo_processed_video_src)

    with col2:
        st.markdown("### Statistics")
        stats = st.session_state.demo_result_stats or {}
        st.metric("Unique Tracks", stats.get("unique_tracks", "N/A"))
        st.metric("Re-ID Matches", stats.get("reid_matches", "N/A"))
        st.metric("Avg FPS", f"{stats.get('avg_fps', 0):.1f}")
        st.metric("Time", f"{stats.get('processing_time', 0):.1f}s")

    st.markdown("---")
    if st.session_state.demo_processed_video_bytes:
        st.download_button(
            label="Download Result",
            data=st.session_state.demo_processed_video_bytes,
            file_name=f"tracked_{st.session_state.demo_result_filename or 'result.mp4'}",
            mime="video/mp4",
            use_container_width=True
        )
