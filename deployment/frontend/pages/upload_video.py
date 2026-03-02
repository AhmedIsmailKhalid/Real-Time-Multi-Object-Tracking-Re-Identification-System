"""
Upload Video Page (with browser-compatible preview)

- Generates a browser-compatible preview MP4 (H.264 + yuv420p + AAC)
- Uploads the ORIGINAL file for processing
- Processing starts ONLY when "Process Video" is clicked
- Clear removes EVERYTHING including the uploaded file by rotating uploader widget key
"""

import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
from utils.api_client import CrossIDClient
from utils.video_utils import get_video_info


# -----------------------------
# Helpers
# -----------------------------
def safe_unlink(path: str | None):
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def make_browser_preview(src_path: str) -> str:
    """
    Create a browser-compatible MP4 preview:
    - Video: H.264 (libx264), yuv420p, baseline for max compatibility
    - Audio: AAC (if present)
    - +faststart so it can begin playing quickly
    Requires: ffmpeg installed and available on PATH.
    """
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cmd_with_audio = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]

    cmd_no_audio = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]

    try:
        subprocess.check_call(cmd_with_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.check_call(cmd_no_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path


# -----------------------------
# Session state
# -----------------------------
if "upload_processing" not in st.session_state:
    st.session_state.upload_processing = False

# One-shot trigger for button click
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False

# Detect new uploads
if "current_upload_id" not in st.session_state:
    st.session_state.current_upload_id = None

if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None

if "preview_video_path" not in st.session_state:
    st.session_state.preview_video_path = None

if "result_video_path" not in st.session_state:
    st.session_state.result_video_path = None

if "last_status" not in st.session_state:
    st.session_state.last_status = None

if "api_client" not in st.session_state:
    st.session_state.api_client = CrossIDClient()

# ✅ This is the key trick to clear the uploader:
# rotate this key to force Streamlit to create a fresh file_uploader widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

client = st.session_state.api_client


# -----------------------------
# Button callbacks
# -----------------------------
def _start_processing():
    st.session_state.start_processing = True


def _clear_all():
    # Delete all temp files
    safe_unlink(st.session_state.get("temp_video_path"))
    safe_unlink(st.session_state.get("preview_video_path"))
    safe_unlink(st.session_state.get("result_video_path"))

    # Reset app state
    st.session_state.temp_video_path = None
    st.session_state.preview_video_path = None
    st.session_state.result_video_path = None
    st.session_state.last_status = None

    st.session_state.upload_processing = False
    st.session_state.start_processing = False
    st.session_state.current_upload_id = None

    # Force file_uploader to clear by changing its key
    st.session_state.uploader_key += 1


# -----------------------------
# Nav / header
# -----------------------------
home_page = st.Page("pages/home.py")
if st.button("← Back to Home"):
    st.switch_page(home_page)

st.markdown("---")
st.title("Upload Video")
st.markdown("Process your own videos with CrossID tracking system")


# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.markdown("### Processing Settings")
    enable_reid = st.checkbox(
        "Enable Re-ID",
        value=True,
        key="upload_enable_reid",
        help="Enable person re-identification",
    )
    show_trails = st.checkbox(
        "Show Trails",
        value=True,
        key="upload_show_trails",
        help="Show movement trails",
    )
    confidence = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="upload_confidence"
    )

    st.markdown("---")
    st.markdown("### Supported Formats")
    st.markdown("- MP4")
    st.markdown("- AVI")
    st.markdown("- MOV")
    st.markdown("")
    st.markdown("**Max size:** 500 MB")


# -----------------------------
# Upload section
# -----------------------------
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"],
    help="Upload a video file to process",
    key=f"video_uploader_{st.session_state.uploader_key}",  # ✅ dynamic key
)

if uploaded_file is not None:
    # Detect new upload (name + size)
    upload_id = f"{uploaded_file.name}-{uploaded_file.size}"
    is_new_upload = st.session_state.current_upload_id != upload_id

    if is_new_upload:
        st.session_state.current_upload_id = upload_id

        # Reset flags so processing never starts accidentally
        st.session_state.upload_processing = False
        st.session_state.start_processing = False
        st.session_state.last_status = None

        # Cleanup old temp files
        safe_unlink(st.session_state.get("temp_video_path"))
        safe_unlink(st.session_state.get("preview_video_path"))
        # Keep result unless you want it cleared on new upload:
        # safe_unlink(st.session_state.get("result_video_path"))
        # st.session_state.result_video_path = None

        # Save uploaded file temporarily
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.temp_video_path = tmp.name

        # Force preview regen
        st.session_state.preview_video_path = None

    temp_path = st.session_state.temp_video_path

    # Show video info
    st.markdown("### Video Information")
    try:
        info = get_video_info(temp_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resolution", f"{info['width']}x{info['height']}")
        col2.metric("FPS", f"{info['fps']:.1f}")
        col3.metric("Frames", info["frame_count"])
        col4.metric("Duration", f"{info['duration']:.1f}s")
    except Exception as e:
        st.warning(f"Could not read video info: {e}")

    # Preview section
    st.markdown("### Preview")
    try:
        pv = st.session_state.get("preview_video_path")
        if not pv or not Path(pv).exists():
            with st.spinner("Preparing preview (transcoding to H.264 for browser compatibility)..."):
                pv = make_browser_preview(temp_path)
            st.session_state.preview_video_path = pv

        st.video(pv, format="video/mp4")
        st.caption("Preview is transcoded for browser compatibility. Original file is used for processing.")
    except FileNotFoundError:
        st.error(
            "ffmpeg is not installed or not on PATH, so preview transcoding failed.\n\n"
            "Install ffmpeg and restart the app."
        )
    except Exception as e:
        st.warning(
            f"Could not generate browser-compatible preview: {e}\n\n"
            "Processing can still work with the original file."
        )
        try:
            st.video(temp_path)
        except Exception:
            pass

    # Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.button(
            "Process Video",
            type="primary",
            use_container_width=True,
            key="process_btn",
            on_click=_start_processing,
        )

    with col2:
        st.button(
            "Clear",
            use_container_width=True,
            key="clear_btn",
            on_click=_clear_all,
        )

    with col3:
        st.caption("Tip: Upload a new file anytime to replace the current one.")


# -----------------------------
# Processing section
# -----------------------------
# Convert one-shot trigger into processing state, then immediately reset trigger
if st.session_state.get("start_processing", False):
    st.session_state.start_processing = False
    st.session_state.upload_processing = True

if st.session_state.get("upload_processing", False):
    st.markdown("---")
    st.markdown("## Processing Video")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if not st.session_state.get("temp_video_path"):
            st.error("No video file found. Please upload a video first.")
            st.session_state.upload_processing = False
            st.stop()

        status_text.text("Uploading to server...")
        job_id = client.upload_video(
            video_path=st.session_state.temp_video_path,  # ORIGINAL file
            enable_reid=enable_reid,
            show_trails=show_trails,
            confidence_threshold=confidence,
        )

        progress_bar.progress(10)
        status_text.text("Video uploaded successfully")
        status_text.text("Processing video...")

        final_status = None

        while True:
            status = client.get_job_status(job_id)

            progress = int(status.get("progress", 0) * 100)
            progress_bar.progress(min(progress, 99))
            status_text.text(f"Processing: {progress}%")

            if status.get("status") == "completed":
                progress_bar.progress(100)
                status_text.text("Processing complete")
                final_status = status
                break

            if status.get("status") == "failed":
                st.error(f"Processing failed: {status.get('message', 'Unknown error')}")
                st.session_state.upload_processing = False
                st.stop()

            time.sleep(0.5)

        status_text.text("Downloading result...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            result_path = tmp.name
            client.download_result(job_id, result_path)

        # Exit processing mode automatically
        st.session_state.upload_processing = False

        # Store result + stats for display outside processing block
        st.session_state.result_video_path = result_path
        st.session_state.last_status = final_status

        status_text.text("")
        st.success("Video processed successfully")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.session_state.upload_processing = False

        if st.button("Try Again"):
            st.session_state.upload_processing = False
            st.rerun()


# -----------------------------
# Results section
# -----------------------------
if st.session_state.get("result_video_path"):
    st.markdown("---")
    st.markdown("## Results")

    result_path = st.session_state.result_video_path
    status = st.session_state.last_status or {}

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Processed Video")
        st.video(result_path, format="video/mp4")

    with col2:
        st.markdown("### Statistics")
        if status.get("stats"):
            stats = status["stats"]
            st.metric("Total Frames", stats.get("total_frames", "N/A"))
            st.metric("Unique Tracks", stats.get("unique_tracks", "N/A"))
            st.metric("Re-ID Matches", stats.get("reid_matches", "N/A"))
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.2f}s")
            st.metric("Average FPS", f"{stats.get('avg_fps', 0):.1f}")
        else:
            st.caption("No stats available.")

    st.markdown("---")
    with open(result_path, "rb") as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name="tracked_output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
