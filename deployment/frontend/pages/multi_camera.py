"""
Multi-Camera Tracking Page
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
    """Convert video to H.264 if needed for browser playback."""
    video_path = Path(video_path)

    cache_dir = Path(tempfile.gettempdir()) / "crossid_preview_cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{video_path.stem}_h264.mp4"
    if cache_file.exists():
        return str(cache_file)

    try:
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
if "mc_processing" not in st.session_state:
    st.session_state.mc_processing = False
if "mc_uploaded_videos" not in st.session_state:
    st.session_state.mc_uploaded_videos = []
if "mc_video_paths" not in st.session_state:
    st.session_state.mc_video_paths = []
if "mc_preview_sources" not in st.session_state:
    st.session_state.mc_preview_sources = []

# Result state
if "mc_result_paths" not in st.session_state:
    st.session_state.mc_result_paths = []
if "mc_result_stats" not in st.session_state:
    st.session_state.mc_result_stats = None
if "mc_result_bytes" not in st.session_state:
    st.session_state.mc_result_bytes = []

# Settings snapshots
if "mc_snapshot_enable_reid" not in st.session_state:
    st.session_state.mc_snapshot_enable_reid = True
if "mc_snapshot_show_trails" not in st.session_state:
    st.session_state.mc_snapshot_show_trails = True
if "mc_snapshot_confidence" not in st.session_state:
    st.session_state.mc_snapshot_confidence = 0.5

# -----------------------------
# Back to home button
# -----------------------------
home_page = st.Page("pages/home.py")
if st.button("← Back to Home"):
    st.switch_page(home_page)

st.markdown("---")

st.title("Multi-Camera Tracking")
st.markdown("Track people across multiple camera views using Re-ID")
st.info("""
**Multi-Camera Re-ID:**
- Upload 2-4 videos from different camera angles
- Videos are processed with a **shared Re-ID gallery**
- Same person gets the **same track ID** across all cameras
- Ideal for monitoring entrances, hallways, parking lots, etc.
""")

# -----------------------------
# Initialize client
# -----------------------------
if "api_client" not in st.session_state:
    st.session_state.api_client = CrossIDClient()
client = st.session_state.api_client

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.markdown("### Processing Settings")
    enable_reid = st.checkbox(
        "Enable Re-ID", value=True, key="mc_enable_reid",
        help="Must be enabled for cross-camera tracking"
    )
    show_trails = st.checkbox("Show Trails", value=True, key="mc_show_trails")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="mc_confidence")

    st.markdown("---")
    st.markdown("### Guidelines")
    st.markdown("- Upload 2-4 videos")
    st.markdown("- Same time period")
    st.markdown("- Different camera angles")
    st.markdown("- MP4, AVI, or MOV")


# -----------------------------
# Helper: render video grid
# -----------------------------
def render_video_grid(video_sources: list, captions: list, use_slots: bool = False):
    """
    Render videos in adaptive grid layout.
    Accepts file paths (str) or bytes.
    Returns list of empty slots if use_slots=True.
    """
    n = len(video_sources)
    slots = []

    def _render_single(idx):
        st.caption(captions[idx])
        if use_slots:
            slot = st.empty()
            slot.video(video_sources[idx])
            slots.append(slot)
        else:
            st.video(video_sources[idx])

    if n == 2:
        cols = st.columns(2)
        for idx in range(2):
            with cols[idx]:
                _render_single(idx)

    elif n == 3:
        cols = st.columns(3)
        for idx in range(3):
            with cols[idx]:
                _render_single(idx)

    else:  # 4 videos - 2x2 grid
        row1 = st.columns(2)
        row2 = st.columns(2)
        for idx in range(2):
            with row1[idx]:
                _render_single(idx)
        for idx in range(2, 4):
            with row2[idx - 2]:
                _render_single(idx)

    return slots if use_slots else None


# -----------------------------
# Upload section
# -----------------------------
if not st.session_state.mc_processing and not st.session_state.mc_result_paths:
    st.markdown("---")
    st.markdown("### Upload Camera Videos")

    uploaded_files = st.file_uploader(
        "Select 2-4 videos from different cameras",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True,
        key="mc_uploader"
    )

    if uploaded_files:
        num_videos = len(uploaded_files)

        if num_videos < 2:
            st.warning("⚠️ Please upload at least 2 videos for multi-camera tracking")
            st.stop()
        elif num_videos > 4:
            st.error("❌ Maximum 4 videos supported. Please remove some videos.")
            st.stop()
        else:
            st.success(f"✓ {num_videos} videos uploaded")

            # Save uploaded files to temp if changed
            current_names = [f.name for f in uploaded_files]
            if st.session_state.mc_uploaded_videos != current_names:
                st.session_state.mc_uploaded_videos = current_names
                st.session_state.mc_video_paths = []
                st.session_state.mc_preview_sources = []

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        st.session_state.mc_video_paths.append(tmp.name)

            # Video info grid
            st.markdown("---")
            st.markdown("### Uploaded Videos")

            cols = st.columns(num_videos) if num_videos <= 3 else st.columns(2)
            for idx, (uploaded_file, video_path) in enumerate(
                zip(uploaded_files, st.session_state.mc_video_paths, strict=False)
            ):
                col_idx = idx if num_videos <= 3 else idx % 2
                with cols[col_idx]:
                    st.markdown(f"**Camera {idx + 1}**")
                    st.caption(uploaded_file.name)
                    try:
                        info = get_video_info(video_path)
                        st.caption(
                            f"{info['width']}x{info['height']} "
                            f"@ {info['fps']:.1f} FPS | "
                            f"{info['duration']:.1f}s"
                        )
                    except Exception as e:
                        st.caption(f"Could not read info: {e}")

            # Preview section
            st.markdown("---")
            st.markdown("### Preview")

            with st.spinner("Preparing browser-compatible previews..."):
                if not st.session_state.mc_preview_sources:
                    st.session_state.mc_preview_sources = [
                        get_browser_compatible_video(path)
                        for path in st.session_state.mc_video_paths
                    ]

            captions = [f"Camera {i + 1}" for i in range(num_videos)]
            render_video_grid(st.session_state.mc_preview_sources, captions)

            # Process / Clear buttons
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("Process All Videos", type="primary", use_container_width=True):
                    if not enable_reid:
                        st.error("❌ Re-ID must be enabled for multi-camera tracking!")
                    else:
                        st.session_state.mc_snapshot_enable_reid = enable_reid
                        st.session_state.mc_snapshot_show_trails = show_trails
                        st.session_state.mc_snapshot_confidence = confidence
                        st.session_state.mc_processing = True
                        st.rerun()

            with col2:
                if st.button("Clear All", use_container_width=True):
                    for path in st.session_state.mc_video_paths:
                        try:
                            Path(path).unlink(missing_ok=True)
                        except Exception:
                            pass
                    st.session_state.mc_uploaded_videos = []
                    st.session_state.mc_video_paths = []
                    st.session_state.mc_preview_sources = []
                    st.rerun()


# -----------------------------
# Processing section
# -----------------------------
if st.session_state.mc_processing:
    st.markdown("---")
    st.markdown("## Processing Multi-Camera Videos")

    num_videos = len(st.session_state.mc_video_paths)
    captions = [f"Camera {i + 1}" for i in range(num_videos)]

    # Show preview grid while processing
    st.markdown("### Camera Views")
    video_slots = render_video_grid(
        st.session_state.mc_preview_sources,
        captions,
        use_slots=True
    )

    # Status section
    st.markdown("---")
    st.markdown("### Processing Status")

    # Overall status
    overall_metric = st.empty()
    overall_progress = st.empty()
    overall_text = st.empty()

    overall_metric.metric("Overall Status", "Uploading")
    overall_bar = overall_progress.progress(0)
    overall_text.write("Starting...")

    # Per-camera status - each in its own container
    st.markdown("#### Per-Camera Status")
    camera_containers = []
    for idx in range(num_videos):
        container = st.container(border=True)
        with container:
            cam_metric = st.empty()
            cam_progress = st.empty()
            cam_metric.metric(f"Camera {idx + 1}", "Queued (0%)")
            cam_progress.progress(0)
        camera_containers.append({
            "metric": cam_metric,
            "progress": cam_progress
        })

    try:
        # Upload all videos
        job_id = client.upload_multi_camera_videos(
            video_paths=st.session_state.mc_video_paths,
            enable_reid=st.session_state.mc_snapshot_enable_reid,
            show_trails=st.session_state.mc_snapshot_show_trails,
            confidence_threshold=st.session_state.mc_snapshot_confidence
        )

        overall_metric.metric("Overall Status", "Processing")

        # Poll for completion
        while True:
            status = client.get_job_status(job_id)

            # Overall progress
            overall = int(status.get("progress", 0) * 100)
            overall_bar.progress(min(overall, 99))
            overall_text.write(f"Overall Progress: {overall}%")

            # Per-camera progress
            video_progress = status.get("video_progress") or [0.0] * num_videos
            video_status = status.get("video_status") or ["queued"] * num_videos

            for idx in range(num_videos):
                cam_status = video_status[idx] if idx < len(video_status) else "queued"
                cam_prog = video_progress[idx] if idx < len(video_progress) else 0.0
                cam_pct = int(cam_prog * 100)

                camera_containers[idx]["metric"].metric(
                    f"Camera {idx + 1}",
                    f"{cam_status.capitalize()} ({cam_pct}%)"
                )
                camera_containers[idx]["progress"].progress(cam_pct)

            if status.get("status") == "completed":
                overall_bar.progress(100)
                overall_metric.metric("Overall Status", "Complete")
                overall_text.write("Overall Progress: 100%")
                for idx in range(num_videos):
                    camera_containers[idx]["metric"].metric(
                        f"Camera {idx + 1}", "Complete (100%)"
                    )
                    camera_containers[idx]["progress"].progress(100)
                break

            elif status.get("status") == "failed":
                overall_metric.metric("Overall Status", "Failed")
                overall_text.write("")
                st.error(f"Processing failed: {status.get('message')}")
                st.session_state.mc_processing = False
                st.stop()

            time.sleep(0.5)

        # Download all results
        overall_text.write("Downloading results...")
        result_paths = []
        result_bytes_list = []

        for camera_id in range(1, num_videos + 1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                result_path = tmp.name
                client.download_multi_camera_result(job_id, camera_id, result_path)
                result_paths.append(result_path)
                with open(result_path, "rb") as f:
                    result_bytes_list.append(f.read())

        st.session_state.mc_result_paths = result_paths
        st.session_state.mc_result_bytes = result_bytes_list
        st.session_state.mc_result_stats = status.get("stats", {})

        # Swap preview slots to processed videos
        for idx, result_path in enumerate(result_paths):
            video_slots[idx].video(result_path)

        st.session_state.mc_processing = False
        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.mc_processing = False


# -----------------------------
# Results section
# -----------------------------
if st.session_state.mc_result_paths and not st.session_state.mc_processing:
    st.markdown("---")
    st.markdown("## Results")

    num_videos = len(st.session_state.mc_result_paths)
    captions = [f"Camera {i + 1}" for i in range(num_videos)]

    # Original preview grid
    st.markdown("### Original Videos")
    render_video_grid(st.session_state.mc_preview_sources, captions)

    st.markdown("---")

    # Processed video grid
    st.markdown("### Processed Videos")
    render_video_grid(st.session_state.mc_result_paths, captions)

    # Overall statistics
    st.markdown("---")
    st.markdown("### Statistics")

    stats = st.session_state.mc_result_stats or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cameras", stats.get("num_cameras", num_videos))
    col2.metric("Unique Tracks", stats.get("unique_tracks", "N/A"))
    col3.metric("Total Re-ID Matches", stats.get("total_reid_matches", "N/A"))
    col4.metric("Total Processing Time", f"{stats.get('total_processing_time', 0):.1f}s")

    # Per-camera statistics in containers
    per_camera = stats.get("per_camera_stats", [])
    if per_camera:
        st.markdown("#### Per-Camera Details")
        cols = st.columns(num_videos)
        for idx, cam_stats in enumerate(per_camera):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**Camera {idx + 1}**")
                    st.metric("Frames", cam_stats.get("total_frames", "N/A"))
                    st.metric("Unique Tracks", cam_stats.get("unique_tracks", "N/A"))
                    st.metric("Re-ID Matches", cam_stats.get("reid_matches", "N/A"))
                    st.metric("Avg FPS", f"{cam_stats.get('avg_fps', 0):.1f}")

    # Download section
    st.markdown("---")
    st.markdown("### Download Results")

    # Merge videos with FFmpeg and offer as single download
    merged_key = "mc_merged_video_bytes"
    if merged_key not in st.session_state:
        st.session_state[merged_key] = None

    if st.session_state[merged_key] is None:
        with st.spinner("Merging all camera views into single video..."):
            try:
                # Build FFmpeg filter for grid layout
                # Creates tiled layout matching the preview grid
                inputs = []
                for path in st.session_state.mc_result_paths:
                    inputs += ['-i', path]

                merged_path = Path(tempfile.gettempdir()) / "crossid_merged_result.mp4"

                if num_videos == 2:
                    # Side by side with labels
                    filter_complex = (
                        "[0:v]drawtext=text='Camera 1':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v0];"
                        "[1:v]drawtext=text='Camera 2':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v1];"
                        "[v0][v1]hstack=inputs=2[v]"
                    )
                elif num_videos == 3:
                    # Three columns with labels
                    filter_complex = (
                        "[0:v]drawtext=text='Camera 1':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v0];"
                        "[1:v]drawtext=text='Camera 2':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v1];"
                        "[2:v]drawtext=text='Camera 3':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v2];"
                        "[v0][v1][v2]hstack=inputs=3[v]"
                    )
                else:
                    # 2x2 grid with labels
                    filter_complex = (
                        "[0:v]drawtext=text='Camera 1':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v0];"
                        "[1:v]drawtext=text='Camera 2':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v1];"
                        "[2:v]drawtext=text='Camera 3':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v2];"
                        "[3:v]drawtext=text='Camera 4':fontsize=36:fontcolor=white"
                        ":box=1:boxcolor=black@0.5:x=10:y=10[v3];"
                        "[v0][v1]hstack=inputs=2[top];"
                        "[v2][v3]hstack=inputs=2[bottom];"
                        "[top][bottom]vstack=inputs=2[v]"
                    )

                cmd = [
                    'ffmpeg',
                    *inputs,
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    str(merged_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0 and merged_path.exists():
                    with open(merged_path, "rb") as f:
                        st.session_state[merged_key] = f.read()
                    st.success("✓ All camera views merged successfully")
                else:
                    st.warning(f"FFmpeg merge failed: {result.stderr[-500:]}")

            except subprocess.TimeoutExpired:
                st.warning("Merge timed out - download individual cameras below")
            except Exception as e:
                st.warning(f"Merge failed: {e} - download individual cameras below")

    # Merged video preview and download
    if st.session_state[merged_key]:
        st.markdown("#### Merged View")
        st.video(st.session_state[merged_key])
        st.download_button(
            label="⬇ Download Merged Video (All Cameras)",
            data=st.session_state[merged_key],
            file_name="crossid_all_cameras_merged.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary"
        )

    # Individual camera downloads
    st.markdown("#### Individual Camera Downloads")
    cols = st.columns(num_videos)
    for idx in range(num_videos):
        with cols[idx]:
            with open(st.session_state.mc_result_paths[idx], "rb") as f:
                st.download_button(
                    label=f"Camera {idx + 1}",
                    data=f,
                    file_name=f"tracked_camera_{idx + 1}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

    # Process new videos button
    st.markdown("---")
    if st.button("Process New Videos", use_container_width=True):
        for path in st.session_state.mc_video_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

        st.session_state.mc_uploaded_videos = []
        st.session_state.mc_video_paths = []
        st.session_state.mc_preview_sources = []
        st.session_state.mc_result_paths = []
        st.session_state.mc_result_bytes = []
        st.session_state.mc_result_stats = None
        st.session_state.mc_processing = False
        st.session_state[merged_key] = None
        st.rerun()
