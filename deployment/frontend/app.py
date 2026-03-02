"""
CrossID - Multi-Camera Object Tracking with Person Re-Identification
Main Streamlit Application
"""

import streamlit as st
from utils.api_client import CrossIDClient

# Page config
st.set_page_config(
    page_title="CrossID - Multi-Object Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages
home_page = st.Page("pages/home.py", title="Home", default=True)
demo_page = st.Page("pages/demo_videos.py", title="Demo Videos")
upload_page = st.Page("pages/upload_video.py", title="Upload Video")
multi_page = st.Page("pages/multi_camera.py", title="Multi-Camera")
search_page = st.Page("pages/person_search.py", title="Person Search")

# Navigation
pg = st.navigation([home_page, demo_page, upload_page, multi_page, search_page])

# Initialize API client globally
if "api_client" not in st.session_state:
    st.session_state.api_client = CrossIDClient()

# Run the selected page
pg.run()
