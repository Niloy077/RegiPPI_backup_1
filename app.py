import streamlit as st
import os
import sys

# --- PATH SETUP FOR SUB-APPS ---
# --- PAGE CONFIG ---
st.set_page_config(page_title="PPI Analyzer", layout="wide")

#RegiPPI
regiPPI_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'regiPPI_app'))
if regiPPI_app_path not in sys.path:
    sys.path.append(regiPPI_app_path)
from regippi_app import run as run_regippi  

# Graph section
graph_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'graph_app'))
if graph_app_path not in sys.path:
    sys.path.append(graph_app_path)
from test_app import run as graph_run  # Assumes test_app.py has a `run()` function

# 3D structure section
structure_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'structure_app'))
if structure_app_path not in sys.path:
    sys.path.append(structure_app_path)
from pdb_to_structure import run as structure_run

# About section
about_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'about_app'))
if about_app_path not in sys.path:
    sys.path.append(about_app_path)
from about_app import run as run_about



# --- SESSION STATE ---
if "active_page" not in st.session_state:
    st.session_state.active_page = "RegiPPI"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #1e1e2f;
            color: white;
        }

        section[data-testid="stSidebar"] [data-testid="baseButton-sidebarCollapseControl"] {
            color: white !important;
        }

        .nav-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .stButton>button {
            width: 100%;
            text-align: left;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background-color: #44446F;
            color: white;
            border: none;
            font-size: 16px;
            transition: background-color 0.3s;
        }

        .stButton>button:hover {
            background-color: #5858ad;
        }

        .stButton>button.active {
            background-color: #7070db !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("logo2.png", use_container_width=True)
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)

    nav_items = ["RegiPPI", "Protein Network", "3D Structure of Protein", "About"]

    for item in nav_items:
        if st.button(item, key=f"nav_{item}"):
            st.session_state.active_page = item

        # Highlight active page
        st.markdown(f"""
            <script>
                const btn = window.parent.document.querySelectorAll('button[kind="secondary"][data-testid="stButton"]');
                const labels = {nav_items};
                const active = "{st.session_state.active_page}";
                btn.forEach((b, i) => {{
                    if (b.innerText.trim() === active) {{
                        b.classList.add("active");
                    }}
                }});
            </script>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN CONTENT ---
if st.session_state.active_page == "RegiPPI":
    run_regippi()

elif st.session_state.active_page == "Protein Network":
    graph_run()

elif st.session_state.active_page == "3D Structure of Protein":
    structure_run()

elif st.session_state.active_page == "About":
    run_about()
