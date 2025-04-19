import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html
import pickle
import requests
import io

def run():
    st.title("🧬 Protein Graph Visualizer")

    @st.cache_data
    def load_data():
        try:
            csv_url = "https://www.dropbox.com/scl/fi/lcbglaqen9nwouc55yckt/new_dataset.csv?rlkey=alidm2r5boqo7qtdyb0e5vxid&st=ub7g08fn&dl=1"
            response_csv = requests.get(csv_url)
            response_csv.raise_for_status()
            ppi_data = pd.read_csv(io.StringIO(response_csv.text))

            pkl_url = "https://drive.google.com/uc?export=download&id=14OQ_urDTncZsskkVPWx20yGwewwn-6_i"
            response_pkl = requests.get(pkl_url)
            response_pkl.raise_for_status()
            embeddings = pickle.load(io.BytesIO(response_pkl.content))

            if ppi_data.empty or not embeddings:
                st.error("Failed to load PPI data or embeddings.")
                return None, None

            st.success(f"✅ Loaded {len(ppi_data)} interactions and {len(embeddings)} embeddings.")
            return ppi_data, embeddings

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None, None

    @st.cache_resource
    def build_graph(ppi_data, embeddings):
        G = nx.Graph()
        for _, row in ppi_data.iterrows():
            G.add_edge(row["protein1"], row["protein2"],
                       weight=row["combined_score"],
                       confidence_level=row["confidence_level"],
                       confidence_level_encoded=row["confidence_level_encoded"],
                       fusion=row["fusion"],
                       cooccurence=row["cooccurence"],
                       homology=row["homology"],
                       coexpression=row["coexpression"],
                       experiments=row["experiments"],
                       database=row["database"],
                       textmining=row["textmining"])
        for protein in G.nodes():
            if protein in embeddings:
                G.nodes[protein]["embedding"] = embeddings[protein]
        return G

    def create_subgraph(graph, selected_proteins, max_neighbors=20):
        nodes = set(selected_proteins)
        for protein in selected_proteins:
            if protein in graph:
                neighbors = sorted(
                    graph[protein].items(),
                    key=lambda x: x[1].get("weight", 0),
                    reverse=True
                )[:max_neighbors]
                for neighbor, _ in neighbors:
                    nodes.add(neighbor)
        return graph.subgraph(nodes).copy()

    def visualize_graph(subgraph, selected_proteins, filename="graph.html"):
        net = Network(height="750px", width="100%", notebook=False)

        for node in subgraph.nodes():
            emb = subgraph.nodes[node].get("embedding", "Not available")
            emb_display = emb[:5] if isinstance(emb, list) else emb
            color = "#eb1e1e" if node in selected_proteins else "#3495eb"
            net.add_node(node, label=node, color=color,
                         title=f"Protein: {node}\nEmbedding (first 5 dims): {emb_display}")

        for edge in subgraph.edges(data=True):
            weight = edge[2]["weight"]
            confidence_level = edge[2]["confidence_level"]
            confidence_level_encoded = edge[2]["confidence_level_encoded"]
            color = "#5cdb56" if confidence_level_encoded == 3 else "#eb9636" if confidence_level_encoded == 2 else "gray"
            scaled_weight = weight * 50
            net.add_edge(edge[0], edge[1], value=scaled_weight, color=color,
                         title=f"Combined Score: {weight}\nConfidence Level: {confidence_level}\nConfidence Encoded: {confidence_level_encoded}\nFusion: {edge[2]['fusion']}\nCooccurence: {edge[2]['cooccurence']}\nHomology: {edge[2]['homology']}\nCoexpression: {edge[2]['coexpression']}\nExperiments: {edge[2]['experiments']}\nDatabase: {edge[2]['database']}\nTextmining: {edge[2]['textmining']}")

        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 100
                }
            },
            "nodes": {
                "shape": "dot",
                "size": 20
            }
        }
        """)

        net.save_graph(filename)
        return filename

    # Start of app logic
    ppi_data, embeddings = load_data()
    if ppi_data is None or embeddings is None:
        st.stop()

    G = build_graph(ppi_data, embeddings)

    if "selected_proteins" not in st.session_state:
        st.session_state.selected_proteins = [ppi_data["protein1"].iloc[0]]

    st.write("Currently exploring:", ", ".join(st.session_state.selected_proteins))

    available_proteins = [p for p in G.nodes() if p not in st.session_state.selected_proteins]
    new_protein = st.selectbox("Add a protein to explore:", ["None"] + available_proteins)
    if new_protein != "None":
        if st.button("Add Protein"):
            st.session_state.selected_proteins.append(new_protein)
            st.rerun()

    if st.button("Reset"):
        st.session_state.selected_proteins = [ppi_data["protein1"].iloc[0]]
        st.rerun()

    if len(st.session_state.selected_proteins) > 1:
        protein_to_remove = st.selectbox("Remove a protein:", st.session_state.selected_proteins)
        if st.button("Remove Protein"):
            st.session_state.selected_proteins.remove(protein_to_remove)
            st.rerun()

    max_neighbors = st.slider("Select number of nearest neighbors per protein:", 5, 15, 5, step=5)

    subgraph = create_subgraph(G, st.session_state.selected_proteins, max_neighbors=max_neighbors)
    html_file = visualize_graph(subgraph, st.session_state.selected_proteins)

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    html(html_content, height=800)

    st.subheader("Embeddings of Selected Proteins (First 5 Dimensions)")
    for protein in st.session_state.selected_proteins:
        emb = G.nodes[protein].get("embedding", "Not available")
        emb_display = emb[:5] if isinstance(emb, list) else emb
        st.write(f"{protein}: {emb_display}")
