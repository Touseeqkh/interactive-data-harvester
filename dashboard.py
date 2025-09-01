import streamlit as st
import pandas as pd
from pyvis.network import Network
import tempfile
import os

st.set_page_config(page_title="Interactive Data Dashboard", layout="wide")

st.title("📊 Interactive Data Harvesting & Network Dashboard")

# --- File Upload ---
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Using default `all_people.csv` if available.")
    try:
        df = pd.read_csv("all_people.csv")
    except FileNotFoundError:
        st.error("No file uploaded and `all_people.csv` not found.")
        st.stop()

# --- Show Data Preview ---
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --- Let user choose columns ---
st.subheader("⚙️ Configure Network Graph")
if df.shape[1] < 2:
    st.error("Need at least 2 columns to create a network graph.")
    st.stop()

source_col = st.selectbox("Select Source Column", options=df.columns)
target_col = st.selectbox("Select Target Column", options=df.columns)

# --- Generate Network Graph ---
if st.button("Generate Network Graph"):
    if source_col not in df.columns or target_col not in df.columns:
        st.error("Invalid columns selected.")
    else:
        st.success(f"Creating graph with `{source_col}` → `{target_col}`")

        net = Network(height="600px", width="100%", notebook=False, bgcolor="#222222", font_color="white")

        for i, row in df.iterrows():
            src = row[source_col]
            tgt = row[target_col]
            if pd.notna(src) and pd.notna(tgt):
                net.add_node(src, label=str(src))
                net.add_node(tgt, label=str(tgt))
                net.add_edge(src, tgt)

        # Save to temporary file and show in Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_file = tmp_file.name

        st.subheader("🌐 Network Graph")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=650, scrolling=True)

        os.unlink(html_file)  # clean up
