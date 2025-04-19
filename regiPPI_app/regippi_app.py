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
    "Protein 1 (9J82)": "https://files.rcsb.org/download/9J82.pdb",
    "Protein 2 (4QQI)": "https://files.rcsb.org/download/4QQI.pdb",
    "Protein 3 (8WRW)": "https://files.rcsb.org/download/8WRW.pdb",
    "Protein 4 (9J0Q)": "https://files.rcsb.org/download/9J0Q.pdb",
    "Protein 5 (6U3V)": "https://files.rcsb.org/download/6U3V.pdb"
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
    st.subheader("Visualize and compare multiple protein interaction. 🚀")
    st.write("Upload your own **PDB files**, or select from our default proteins (max 5 total).")

    # Upload PDBs
    uploaded_files = st.file_uploader("📂 Upload PDB Files", type=["pdb"], accept_multiple_files=True)
    uploaded_files = uploaded_files or []

    uploaded_names = set()
    cleaned_uploaded_files = []
    for file in uploaded_files:
        name = file.name.replace(".pdb", "")
        if name not in uploaded_names:
            uploaded_names.add(name)
            cleaned_uploaded_files.append(file)

    # Select default PDBs
    use_default = st.checkbox("Use default PDB files")
    selected_pdbs = []
    fetched_files = []

    if use_default:
        available_default_names = [name for name in DEFAULT_PDB_FILES if name.replace(" (", "_").split("_")[0] not in uploaded_names]
        selected_pdbs = st.multiselect("Select default proteins:", available_default_names)

        for pdb_name in selected_pdbs:
            pdb_io = fetch_pdb_from_url(DEFAULT_PDB_FILES[pdb_name])
            if pdb_io:
                fetched_files.append((pdb_io, pdb_name))

    # Combine both sets, enforce max limit = 5
    final_uploaded_files = []

    for file in cleaned_uploaded_files:
        final_uploaded_files.append(file)

    for file, name in fetched_files:
        if len(final_uploaded_files) < 5:
            final_uploaded_files.append((file, name))
        else:
            st.warning("You can only compare up to 5 proteins at once.")
            break

    if len(final_uploaded_files) > 5:
        st.warning("Too many files selected. Please limit to 5 proteins total.")
        return

    if final_uploaded_files:
        embeddings_list = []
        protein_names = []

        for file in final_uploaded_files:
            if isinstance(file, tuple):  # Default
                pdb_io, name = file
                protein_names.append(name)
            else:  # Uploaded
                pdb_content = file.read()
                if isinstance(pdb_content, bytes):
                    pdb_content = pdb_content.decode("latin-1")
                pdb_io = io.StringIO(pdb_content)
                protein_names.append(file.name.replace(".pdb", ""))

            # Extract sequence & get embedding
            seq = extract_sequence_from_pdb(pdb_io)
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings_list.append(embedding)

        # Similarity & Visualization
        embedding_matrix = np.vstack(embeddings_list)
        embedding_2d = embedding_matrix[:, :2] if embedding_matrix.shape[0] > 1 else np.hstack((embedding_matrix, np.zeros((1, 1))))
        similarity_matrix = cosine_similarity(embedding_matrix)

        interaction_threshold = 0.7

        st.info("""
        ### 🔎 Visualization Guide:
        - 🔵 **Blue circles** represent proteins in the embedding space.
        - 🟢 **Green lines** indicate predicted **interactions** between proteins.
        - ✅ If two proteins are connected with a green line, their similarity is **≥ 0.7**, indicating interaction.
        - ❌ If there is **no line**, the proteins are predicted to **not interact** under the current threshold.
        """)

        fig, ax = plt.subplots()
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', s=100)

        for i, name in enumerate(protein_names):
            ax.text(embedding_2d[i, 0], embedding_2d[i, 1], name, fontsize=12, ha="right")

        for i in range(len(protein_names)):
            for j in range(i + 1, len(protein_names)):
                sim = similarity_matrix[i, j]
                if sim >= interaction_threshold:
                    x1, y1 = embedding_2d[i]
                    x2, y2 = embedding_2d[j]
                    ax.plot([x1, x2], [y1, y2], 'g-', alpha=sim)
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{sim:.2f}", fontsize=10, color='darkgreen', ha='center',
                            bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Protein Embedding Comparison with Interactions")
        st.pyplot(fig)
