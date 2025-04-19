import streamlit as st
from PIL import Image
import base64
import os

# --- Helper to convert image to base64 ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Main run function ---
def run():
     # Must be the FIRST Streamlit command

    st.header("📘 About REGi-PPI")

    # --- TEAM MEMBERS ---
    st.subheader("👥 Meet the Team")
    team_images = ["team1.jpg", "team2.jpg", "team3.jpg", "team4.jpg"]
    team_names = ["Tausif Mushtaque", "Niloy Biswas", "Nandini Das", "Syed Riaz"]

    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            img_path = os.path.join(os.path.dirname(__file__), team_images[i])
            if os.path.exists(img_path):
                img_base64 = get_base64_image(img_path)
                st.markdown(
                    f"""
                    <div style="text-align:center">
                        <img src="data:image/jpeg;base64,{img_base64}" 
                             style="border-radius: 50%; width: 150px; height: 150px; object-fit: cover; margin-bottom: 10px;" />
                        <div style="font-weight: bold;">{team_names[i]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Image {team_images[i]} not found!")

    st.markdown("---")
    st.subheader("📊 System Diagram & Methodology")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        REGi-PPI works by predicting edges between proteins using embeddings and GNN-based regression:
        
        - **Data Source**: STRING database (interactions filtered with score ≥ 400)
        - **Embedding**: Protein sequences are encoded using **ProtBERT**
        - **Graph Construction**:
            - Nodes: Protein embeddings
            - Edges: Combined embeddings + STRING score
        - **GNN Models**: Trained using **GCN** and **GAT**
        - **Output**: Interaction probability & combined score of new protein
        
        The model enables edge-level prediction and regression — predicting both presence and interaction strength.
        """)
    with col2:
        image_path = os.path.join(os.path.dirname(__file__), "model_image.png")
        if os.path.exists(image_path):
            st.image(image_path, caption="Figure: REGi-PPI Workflow", use_container_width=True)
        else:
            st.warning("Workflow image not found!")

    st.markdown("---")
    st.subheader("🚀 Novelty of REGi-PPI")
    st.markdown("""
    While models like **Struct2Graph** handle edge classification, REGi-PPI introduces regression to learn edge weights and return quantified scores of protein interaction.  
    The method is lightweight, simple, and flexible—allowing rapid predictions using pre-trained embeddings and GNNs.
    """)

    st.markdown("---")
    st.subheader("🌍 Societal & Environmental Impact")
    st.markdown("""
    - **Healthcare**: Enables faster diagnosis and treatment discovery via PPI mapping  
    - **Sustainability**: Reduces lab waste and power usage through computational prediction  
    - **Ethics**: Bypasses unnecessary experiments, encouraging data-driven safe research  
    """)

    st.markdown("---")
    st.subheader("💼 Business Model & Future Scope")
    st.markdown("""
    The model serves as a **prototype** for scalable microbiological deep learning applications.  
    Future goals include:
    - Setting up a research center focused on GNNs in bioinformatics
    - Scaling models using **high-performance compute clusters**
    - Partnering with institutions that manage sensitive biological data  
    Inspired by tools like **AlphaFold-3**, REGi-PPI could shape the next era of AI-based microbiological discovery.
    """)

    st.markdown("---")
    st.subheader("🧪 Results")
    col1, col2, col3 = st.columns(3)
    for col, name in zip([col1, col2, col3], ["result1.png", "result2.png", "result3.png"]):
        img_path = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(img_path):
            col.image(img_path, caption=name.replace(".png", "").capitalize(), use_container_width=True)
        else:
            col.warning(f"{name} not found!")

    st.markdown("---")
    st.subheader("📌 Conclusion")
    st.markdown("""
    REGi-PPI was trained as a **CPU-based prototype** with simplified subgraph sampling.  
    Planned upgrades:
    - Extend to multiple **benchmark datasets**
    - Experiment with **advanced sampling strategies**
    - Build a **custom GNN architecture**
    - Package the app using **Docker** and **Hydra** for portability across devices
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <strong>© 2025 REGi-PPI Team</strong><br>
    </div>
    """, unsafe_allow_html=True)
