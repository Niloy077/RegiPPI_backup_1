import streamlit as st
import py3Dmol
import requests

# Default PDB options
DEFAULT_PDB_FILES = {
    "Protein 1 (9J82)": "https://files.rcsb.org/download/9J82.pdb",
    "Protein 2 (4QQI)": "https://files.rcsb.org/download/4QQI.pdb",
    "Protein 3 (8WRW)": "https://files.rcsb.org/download/8WRW.pdb",
    "Protein 4 (9J0Q)": "https://files.rcsb.org/download/9J0Q.pdb",
    "Protein 5 (6U3V)": "https://files.rcsb.org/download/6U3V.pdb",
    "Protein 6 (8D41)": "https://files.rcsb.org/download/8D41.pdb",
    "Protein 7 (2B6H)": "https://files.rcsb.org/download/2B6H.pdb",
    "Protein 8 (4UJ4)": "https://files.rcsb.org/download/4UJ4.pdb",
    "Protein 9 (5H3D)": "https://files.rcsb.org/download/5H3D.pdb"
}
def run():
    # --- Streamlit Layout ---
    st.title("🧬 PDB to 3D Protein Structure Viewer")

    col1, col2 = st.columns([1, 2])  # Left and Right

    # --- Left Column: Input Section ---
    with col1:
        st.header("Input Options")

        # Default PDB selector
        selected_protein = st.selectbox("Choose a default protein", list(DEFAULT_PDB_FILES.keys()))
        default_url = DEFAULT_PDB_FILES[selected_protein]

        # Upload PDB file from computer
        uploaded_pdb = st.file_uploader("Or upload a PDB file", type=["pdb"])

    # --- Get PDB data ---
    if uploaded_pdb is not None:
        pdb_data = uploaded_pdb.read().decode("utf-8")
    else:
        try:
            response = requests.get(default_url)
            pdb_data = response.text
        except:
            pdb_data = ""
            st.error("Failed to load the PDB file.")

    # --- Right Column: 3D Viewer ---
    with col2:
        st.header("🧪 3D Protein Structure")
        if pdb_data:
            view = py3Dmol.view(width=400, height=400)
            view.addModelsAsFrames(pdb_data)
            view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
            view.zoomTo()
            view_html = view._make_html()

            # Wrap the entire viewer in a bordered container directly in the HTML
            full_html = f"""
            <div style="border: 3px solid black; border-radius: 10px; padding: 10px; display: inline-block;">
                {view_html}
            </div>
            """
            st.components.v1.html(full_html, height=440, width=640)  # Slightly increased to fit the border

        else:
            st.warning("No PDB data to display.")
