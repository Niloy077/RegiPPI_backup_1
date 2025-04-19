# regippi_app.py
import streamlit as st
from Bio import PDB
from Bio.SeqUtils import seq1
from transformers import EsmModel, EsmTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load ESM-2 Model & Tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Extract sequence from PDB
def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    seq += seq1(residue.get_resname())
    return seq

# Default PDB files (URLs)
DEFAULT_PDB_FILES = {
    "PROTEIN 1 (9J82)": "https://files.rcsb.org/download/9J82.pdb",
    "PROTEIN 2 (4QQI)": "https://files.rcsb.org/download/4QQI.pdb",
    "PROTEIN 3 (8WRW)": "https://files.rcsb.org/download/8WRW.pdb",
    "PROTEIN 4 (9J0Q)": "https://files.rcsb.org/download/9J0Q.pdb",
    "PROTEIN 5 (6U3V)": "https://files.rcsb.org/download/6U3V.pdb"
}

# Fetch PDB file from URL
def fetch_pdb_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return io.StringIO(response.text)
    else:
        st.error(f"Failed to fetch PDB from {url}")
        return None

# Main Run Function
def run():
    st.title("🔬 Protein-Protein Interaction Visualizer")
    st.subheader("Visualize and compare multiple protein interactions. 🚀")
    st.write("Upload or choose up to 5 **PDB files** at a time. Avoid duplicates by name.")

    # Session state to store selected protein info
    if 'all_proteins' not in st.session_state:
        st.session_state.all_proteins = {}
    
    uploaded_files = st.file_uploader("📂 Upload PDB Files (max 5 at once)", type=["pdb"], accept_multiple_files=True)
    use_default = st.checkbox("Use default PDB files")

    selected_files = []

    if use_default:
        max_remaining = max(0, 5 - len(st.session_state.all_proteins))
        options = [name for name in DEFAULT_PDB_FILES if name.upper() not in st.session_state.all_proteins]
        selected_defaults = st.multiselect("Select default proteins:", options, max_selections=max_remaining)
        for name in selected_defaults:
            fetched = fetch_pdb_from_url(DEFAULT_PDB_FILES[name])
            if fetched:
                selected_files.append((fetched, name))

    if uploaded_files:
        for file in uploaded_files[:5 - len(selected_files)]:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("latin-1")
            name = file.name.upper().replace(".PDB", "")
            if name not in st.session_state.all_proteins:
                selected_files.append((io.StringIO(content), name))

    # Process only new proteins
    if selected_files:
        for pdb_io, name in selected_files:
            if name.upper() in st.session_state.all_proteins:
                continue
            try:
                seq = extract_sequence_from_pdb(pdb_io)
                inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                st.session_state.all_proteins[name.upper()] = embedding
            except Exception as e:
                st.warning(f"Failed to process {name}: {str(e)}")

    # Plot if more than 1
    if len(st.session_state.all_proteins) > 1:
        st.success(f"{len(st.session_state.all_proteins)} unique proteins loaded.")
        protein_names = list(st.session_state.all_proteins.keys())
        embedding_matrix = np.vstack([st.session_state.all_proteins[name] for name in protein_names])
        embedding_2d = embedding_matrix[:, :2]
        similarity_matrix = cosine_similarity(embedding_matrix)
        interaction_threshold = 0.7

        st.info("""
        ### 🔎 Visualization Guide:
        - 🔵 **Blue circles** = Proteins.
        - 🟢 **Green lines** = Similarity ≥ 0.7 (likely interaction).
        """)

        fig, ax = plt.subplots()
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', s=100)
        for i, name in enumerate(protein_names):
            ax.text(embedding_2d[i, 0], embedding_2d[i, 1], name, fontsize=10, ha="right")

        for i in range(len(protein_names)):
            for j in range(i + 1, len(protein_names)):
                sim = similarity_matrix[i, j]
                if sim >= interaction_threshold:
                    x1, y1 = embedding_2d[i]
                    x2, y2 = embedding_2d[j]
                    ax.plot([x1, x2], [y1, y2], 'g-', alpha=min(sim, 0.99))
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{sim:.2f}", fontsize=8, color='darkgreen',
                            ha='center', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Protein Embedding Comparison")
        st.pyplot(fig)

    # Option to clear
    if st.button("🔄 Clear All Proteins"):
        st.session_state.all_proteins = {}

# Call the run function
run()
