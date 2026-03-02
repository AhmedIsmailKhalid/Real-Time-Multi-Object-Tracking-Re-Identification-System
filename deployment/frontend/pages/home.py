"""
Home Page
"""

import streamlit as st
from utils.api_client import CrossIDClient  # noqa: F401

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    section[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">CrossID</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Camera Object Tracking with Person Re-Identification</p>',
    unsafe_allow_html=True
)

# Check backend health
try:
    client = st.session_state.api_client
    health = client.health_check()

    with st.sidebar:
        st.success("Backend Connected")
        st.json({
            "Status": health["status"],
            "Device": health["device"],
            "CUDA": "Available" if health["cuda_available"] else "Not Available",
            "Models": "Loaded" if health["models_loaded"] else "Not Loaded"
        })

except Exception as e:
    st.error(f"Backend connection failed: {e}")
    st.info("Make sure FastAPI backend is running")
    st.stop()

# Main content
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

demo_page = st.Page("pages/demo_videos.py")
upload_page = st.Page("pages/upload_video.py")
multi_page = st.Page("pages/multi_camera.py")
search_page = st.Page("pages/person_search.py")

with col1:
    st.markdown("### Demo Videos")
    st.write("Try pre-loaded demo scenarios")
    if st.button("Go to Demos", use_container_width=True):
        st.switch_page(demo_page)

with col2:
    st.markdown("### Upload Video")
    st.write("Process your own videos")
    if st.button("Upload", use_container_width=True):
        st.switch_page(upload_page)

with col3:
    st.markdown("### Multi-Camera")
    st.write("Re-ID across cameras")
    if st.button("View Demo", use_container_width=True):
        st.switch_page(multi_page)

with col4:
    st.markdown("### Person Search")
    st.write("Find person in video")
    if st.button("Search", use_container_width=True):
        st.switch_page(search_page)

# Features
st.markdown("---")
st.markdown("## Key Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    **Real-Time Tracking**
    - 30+ FPS on GPU
    - YOLOv8 detection
    - ByteTrack algorithm
    """)

with feat_col2:
    st.markdown("""
    **Person Re-ID**
    - 99% Rank-1 accuracy
    - 70% mAP on Market-1501
    - Cross-camera matching
    """)

with feat_col3:
    st.markdown("""
    **Advanced Features**
    - Movement trails
    - Track statistics
    - Export results
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Built using FastAPI, Streamlit, YOLOv8, and PyTorch"
    "</p>",
    unsafe_allow_html=True
)
