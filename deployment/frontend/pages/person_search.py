"""
Person Search Page
"""

import io
import json
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image
from utils.api_client import CrossIDClient

# Back to home button
home_page = st.Page("pages/home.py")
if st.button("← Back to Home"):
    st.switch_page(home_page)

st.markdown("---")

st.title("Person Search")
st.markdown("Search for a specific person across videos using Re-ID")

st.info("""
**How it works:**
1. **Upload videos** to build a searchable database (or use previously processed videos)
2. Upload a **cropped image** of the person to find
3. System searches across selected videos and returns matches

**Note:** Currently requires pre-cropped images. Auto-detection coming soon!
""")

# Initialize client
if "api_client" not in st.session_state:
    st.session_state.api_client = CrossIDClient()
client = st.session_state.api_client

# Session state
if "ps_uploaded_videos" not in st.session_state:
    st.session_state.ps_uploaded_videos = []
if "ps_video_paths" not in st.session_state:
    st.session_state.ps_video_paths = []
if "ps_processing_status" not in st.session_state:
    st.session_state.ps_processing_status = {}
if "ps_job_ids" not in st.session_state:
    st.session_state.ps_job_ids = []
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "query_image_path" not in st.session_state:
    st.session_state.query_image_path = None

# Sidebar settings
with st.sidebar:
    st.markdown("### Processing Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="ps_confidence")
    st.markdown("---")
    st.markdown("### Search Scope")
    search_uploaded = st.checkbox("Search uploaded videos", value=True, key="ps_search_uploaded",
                                   help="Search videos uploaded via Upload Video page")
    search_multicam = st.checkbox("Search multi-camera sessions", value=True, key="ps_search_multicam",
                                   help="Search multi-camera processed videos")
    search_local = st.checkbox("Search videos on this page", value=True, key="ps_search_local",
                               help="Search videos uploaded on Person Search page")

# Video upload section
st.markdown("---")
st.markdown("## Step 1: Upload Videos (Optional)")
st.markdown("Upload videos to build your searchable database, or skip if you've already processed videos elsewhere.")

uploaded_files = st.file_uploader(
    "Upload videos to search through",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True,
    key="ps_video_uploader",
    help="Upload videos containing people you want to search for"
)

if uploaded_files:
    # Save uploaded files to temp
    current_names = [f.name for f in uploaded_files]
    if st.session_state.ps_uploaded_videos != current_names:
        st.session_state.ps_uploaded_videos = current_names
        st.session_state.ps_video_paths = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.ps_video_paths.append(tmp.name)

    st.success(f"✓ {len(uploaded_files)} video(s) ready for processing")

    # Show video list
    with st.expander("📹 Uploaded Videos", expanded=False):
        for idx, name in enumerate(st.session_state.ps_uploaded_videos):
            status = st.session_state.ps_processing_status.get(name, "Not processed")
            st.markdown(f"{idx+1}. **{name}** - {status}")

    # Process videos button
    if st.button("Process Videos for Search", type="primary", use_container_width=True):
        st.markdown("### Processing Videos...")

        for _idx, (video_name, video_path) in enumerate(zip(st.session_state.ps_uploaded_videos,
                                                            st.session_state.ps_video_paths, strict=False)):
            if st.session_state.ps_processing_status.get(video_name) == "✓ Processed":
                continue

            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            status_placeholder.info(f"Processing {video_name}...")

            try:
                # Upload video
                job_id = client.upload_video(
                    video_path=video_path,
                    enable_reid=True,  # Must enable Re-ID for search
                    show_trails=True,
                    confidence_threshold=confidence
                )

                st.session_state.ps_job_ids.append(job_id)

                # Poll for completion
                progress_bar = progress_placeholder.progress(0)

                while True:
                    status = client.get_job_status(job_id)
                    progress = int(status.get("progress", 0) * 100)
                    progress_bar.progress(min(progress, 99))

                    if status.get("status") == "completed":
                        progress_bar.progress(100)
                        st.session_state.ps_processing_status[video_name] = "✓ Processed"
                        status_placeholder.success(f"✓ {video_name} processed successfully")
                        break
                    elif status.get("status") == "failed":
                        st.session_state.ps_processing_status[video_name] = "✗ Failed"
                        status_placeholder.error(f"✗ {video_name} failed: {status.get('message')}")
                        break

                    time.sleep(0.5)

            except Exception as e:
                st.session_state.ps_processing_status[video_name] = f"✗ Error: {str(e)}"
                status_placeholder.error(f"✗ Error processing {video_name}: {e}")

        st.success("All videos processed! Ready to search.")
        st.rerun()

# Query image section
st.markdown("---")
st.markdown("## Step 2: Upload Query Image")
st.markdown("Upload a cropped image of the person you want to find.")

col1, col2 = st.columns([2, 1])

with col1:
    query_file = st.file_uploader(
        "Choose an image of the person to search",
        type=["jpg", "jpeg", "png"],
        key="ps_query_uploader",
        help="Upload a cropped image showing the person's face and upper body"
    )

with col2:
    st.markdown("### Guidelines")
    st.markdown("✓ Single person only")
    st.markdown("✓ Clear face visible")
    st.markdown("✓ Upper body included")
    st.markdown("✓ Good lighting")

# Preview query image
if query_file is not None:
    st.markdown("---")
    st.markdown("### Query Image Preview")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(query_file.name).suffix) as tmp:
        tmp.write(query_file.getbuffer())
        st.session_state.query_image_path = tmp.name

    # Display preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(st.session_state.query_image_path)
        st.image(image, caption="Query Person", use_container_width=True)
        st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")

# Search settings
st.markdown("---")
st.markdown("## Step 3: Search Settings")

col1, col2 = st.columns(2)

with col1:
    similarity_threshold = st.slider(
        "Match Threshold",
        0.0, 1.0, 0.85, 0.05,
        key="ps_similarity",
        help="Minimum similarity score (0-1). Higher = stricter matching."
    )
    st.caption("**Recommended:** 0.85 for high confidence, 0.75 for more results")

with col2:
    max_results = st.number_input(
        "Max Results",
        min_value=1,
        max_value=200,
        value=50,
        key="ps_max_results",
        help="Maximum number of matches to return"
    )

# Search button
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    search_button = st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.query_image_path is None)
    )

with col2:
    if st.button("Clear Results", use_container_width=True):
        st.session_state.search_results = None
        st.rerun()

# Perform search
if search_button:
    # Validate search scope
    if not any([search_uploaded, search_multicam, search_local]):
        st.warning("⚠️ Please select at least one search scope in the sidebar")
    else:
        with st.spinner("Searching across selected videos..."):
            try:
                results = client.search_person(
                    query_image_path=st.session_state.query_image_path,
                    similarity_threshold=similarity_threshold,
                    max_results=max_results
                )

                # Filter results based on search scope
                matches = results.get("matches", [])
                filtered_matches = []

                for match in matches:
                    job_id = match["job_id"]
                    video_name = match["video_filename"]

                    # Determine video source
                    include = False

                    if search_local and job_id in st.session_state.ps_job_ids:
                        include = True
                    elif search_multicam and "multi_camera" in video_name:
                        include = True
                    elif search_uploaded and job_id not in st.session_state.ps_job_ids and "multi_camera" not in video_name:
                        include = True

                    if include:
                        filtered_matches.append(match)

                results["matches"] = filtered_matches
                results["total_matches"] = len(filtered_matches)

                st.session_state.search_results = results
                st.rerun()

            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = None

# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## Search Results")

    results = st.session_state.search_results
    matches = results.get("matches", [])

    if not matches:
        st.warning("No matches found. Try:\n- Lowering the similarity threshold\n- Processing more videos\n- Expanding search scope in sidebar")
    else:
        st.success(f"Found {len(matches)} match(es) across selected videos!")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(matches))
        col2.metric("Threshold Used", f"{results.get('threshold', 0.85):.2f}")
        col3.metric("Avg Similarity", f"{sum(m['similarity'] for m in matches) / len(matches):.3f}")

        st.markdown("---")

        # Results table
        st.markdown("### Matches")

        for idx, match in enumerate(matches):
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    st.markdown(f"**Match {idx + 1}**")
                    st.caption(f"Video: {match['video_filename']}")
                    st.caption(f"Job ID: `{match['job_id'][:8]}...`")

                with col2:
                    st.metric("Track ID", match['track_id'])

                with col3:
                    similarity_pct = match['similarity'] * 100
                    st.metric("Confidence", f"{similarity_pct:.1f}%")

                with col4:
                    st.metric("Detections", match['num_detections'])

        # Export options
        st.markdown("---")
        st.markdown("### Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV export
            csv_buffer = io.StringIO()
            csv_buffer.write("Match,Video,Job_ID,Track_ID,Similarity,Detections\n")
            for idx, match in enumerate(matches):
                csv_buffer.write(
                    f"{idx+1},{match['video_filename']},{match['job_id']},"
                    f"{match['track_id']},{match['similarity']},{match['num_detections']}\n"
                )

            st.download_button(
                label="📄 CSV",
                data=csv_buffer.getvalue(),
                file_name="person_search_results.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download as spreadsheet-compatible CSV"
            )

        with col2:
            # JSON export with full details
            json_data = {
                "search_metadata": {
                    "timestamp": time.time(),
                    "search_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "threshold": results.get('threshold', 0.85),
                    "total_matches": len(matches),
                    "query_image": query_file.name if query_file else "unknown",
                    "average_confidence": sum(m['similarity'] for m in matches) / len(matches)
                },
                "matches": matches
            }

            st.download_button(
                label="📋 JSON",
                data=json.dumps(json_data, indent=2),
                file_name="person_search_results.json",
                mime="application/json",
                use_container_width=True,
                help="Download as structured JSON data"
            )

        with col3:
            # Detailed text report
            report_buffer = io.StringIO()
            report_buffer.write("=" * 70 + "\n")
            report_buffer.write("PERSON SEARCH REPORT\n")
            report_buffer.write("=" * 70 + "\n\n")
            report_buffer.write(f"Query Image: {query_file.name if query_file else 'N/A'}\n")
            report_buffer.write(f"Search Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_buffer.write(f"Similarity Threshold: {results.get('threshold', 0.85):.2f}\n")
            report_buffer.write(f"Total Matches: {len(matches)}\n")
            report_buffer.write(f"Average Confidence: {sum(m['similarity'] for m in matches) / len(matches):.3f}\n")
            report_buffer.write(f"Confidence Range: {min(m['similarity'] for m in matches):.3f} - {max(m['similarity'] for m in matches):.3f}\n")
            report_buffer.write("\n" + "=" * 70 + "\n")
            report_buffer.write("SEARCH SCOPE\n")
            report_buffer.write("=" * 70 + "\n\n")
            report_buffer.write(f"  • Search uploaded videos: {search_uploaded}\n")
            report_buffer.write(f"  • Search multi-camera sessions: {search_multicam}\n")
            report_buffer.write(f"  • Search videos on this page: {search_local}\n")
            report_buffer.write("\n" + "=" * 70 + "\n")
            report_buffer.write("DETAILED RESULTS\n")
            report_buffer.write("=" * 70 + "\n\n")

            for idx, match in enumerate(matches):
                report_buffer.write(f"Match #{idx + 1}\n")
                report_buffer.write(f"  Video: {match['video_filename']}\n")
                report_buffer.write(f"  Job ID: {match['job_id']}\n")
                report_buffer.write(f"  Track ID: {match['track_id']}\n")
                report_buffer.write(f"  Confidence: {match['similarity'] * 100:.1f}%\n")
                report_buffer.write(f"  Detections: {match['num_detections']}\n")
                report_buffer.write(f"  {'-' * 68}\n\n")

            report_buffer.write("=" * 70 + "\n")
            report_buffer.write("END OF REPORT\n")
            report_buffer.write("=" * 70 + "\n")

            st.download_button(
                label="📊 Report",
                data=report_buffer.getvalue(),
                file_name="person_search_report.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download detailed formatted report"
            )

# Help section
st.markdown("---")
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    **Workflow:**
    1. Upload videos on this page OR process videos via "Upload Video" page first
    2. Upload a clear, cropped image of the person
    3. Adjust search scope in sidebar (which videos to search)
    4. Click Search

    **Image Quality:**
    - Use well-lit, clear photos
    - Include face + upper body (not just face)
    - Avoid extreme angles or occlusions

    **Threshold Settings:**
    - **0.90-1.00:** Very strict (exact matches only)
    - **0.85-0.89:** Recommended (high confidence)
    - **0.75-0.84:** Relaxed (more results, some false positives)
    - **Below 0.75:** Very relaxed (many false positives)

    **Search Scope:**
    - Use sidebar checkboxes to control which videos are searched
    - Uncheck unnecessary sources to speed up search

    **Export Options:**
    - **CSV:** For Excel/Google Sheets analysis
    - **JSON:** For programmatic processing
    - **Report:** Human-readable summary with full details

    **Troubleshooting:**
    - No results? Lower threshold or ensure videos were processed with Re-ID enabled
    - Too many false positives? Raise the threshold
    - Videos not showing up? Check search scope checkboxes in sidebar
    """)
