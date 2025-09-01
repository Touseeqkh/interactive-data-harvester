# dashboard.py
import streamlit as st
import pandas as pd
from pyvis.network import Network
import tempfile
import os

st.set_page_config(page_title="Interactive Data Harvesting & Network Dashboard", layout="wide")
st.title("📊 Interactive Data Harvesting & Network Dashboard")

# === 1. Data Upload / Harvesting ===
st.sidebar.header("1️⃣ Upload / Harvest Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
else:
    st.info("No file uploaded. Using default `all_people.csv` if available.")
    try:
        df = pd.read_csv("all_people.csv")
    except FileNotFoundError:
        st.error("No file uploaded and default `all_people.csv` not found.")
        st.stop()

# --- Preview raw data ---
st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head(10))

# === 2. Data Cleaning Options ===
st.sidebar.header("2️⃣ Data Cleaning Options")
drop_dupes = st.sidebar.checkbox("Remove duplicates", value=True)
missing_action = st.sidebar.radio(
    "Handle missing values",
    ["Do nothing", "Drop rows", "Fill with placeholder"],
    index=0
)
normalize_text = st.sidebar.checkbox("Normalize text columns (lowercase & strip)")
exclude_rows = st.sidebar.checkbox("Add 'Exclude' column for manual filtering")

# --- Apply Cleaning ---
df_clean = df.copy()
if drop_dupes:
    df_clean = df_clean.drop_duplicates()

if missing_action == "Drop rows":
    df_clean = df_clean.dropna()
elif missing_action == "Fill with placeholder":
    df_clean = df_clean.fillna("MISSING")

if normalize_text:
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

if exclude_rows:
    df_clean["Exclude"] = False
    st.info("You can edit this 'Exclude' column in Excel/Google Sheets after export.")

# --- Preview cleaned data ---
st.subheader("✨ Cleaned Data Preview")
st.dataframe(df_clean.head(10))
st.write(f"Original shape: {df.shape} → Cleaned shape: {df_clean.shape}")

# --- Download cleaned CSV ---
st.download_button(
    "💾 Download Cleaned CSV",
    df_clean.to_csv(index=False).encode("utf-8"),
    "cleaned_data.csv",
    "text/csv"
)

# === 3. Network Graph ===
st.sidebar.header("3️⃣ Network Graph Options")
cols = df_clean.columns.tolist()
if len(cols) < 2:
    st.warning("Need at least 2 columns to build a network graph.")
else:
    source_col = st.sidebar.selectbox("Select Source Column", cols)
    target_col = st.sidebar.selectbox("Select Target Column", cols)

    if st.sidebar.button("Generate Network Graph"):
        st.success(f"Building graph using `{source_col}` → `{target_col}`")

        net = Network(height="600px", width="100%", notebook=False, bgcolor="#222222", font_color="white")

        for i, row in df_clean.iterrows():
            src = row[source_col]
            tgt = row[target_col]
            if pd.notna(src) and pd.notna(tgt):
                net.add_node(src, label=str(src))
                net.add_node(tgt, label=str(tgt))
                net.add_edge(src, tgt)

        # Save to temporary HTML file and display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_file = tmp_file.name

        st.subheader("🌐 Network Graph")
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=650, scrolling=True)

        os.unlink(html_file)

# === 4. Search Person Connections ===
st.sidebar.header("4️⃣ Search Person Connections")
if 'Exclude' in df_clean.columns:
    df_searchable = df_clean[df_clean['Exclude'] == False]
else:
    df_searchable = df_clean

if source_col in df_searchable.columns and target_col in df_searchable.columns:
    person = st.sidebar.selectbox("Select a person to see connections:", df_searchable[source_col].unique())
    connections = df_searchable[df_searchable[source_col] == person][target_col].tolist()
    st.subheader(f"🔍 Connections for {person}")
    st.write(connections)
