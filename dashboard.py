# dashboard.py
import streamlit as st
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

st.set_page_config(page_title="Data Harvesting & Network Dashboard", layout="wide")
st.title("📊 Data Harvesting & Network Dashboard")

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"

# ============ UTILITIES ============
def is_human(title):
    """Check if a Wikipedia page corresponds to a human using Wikidata"""
    try:
        # Get Wikidata ID
        r = requests.get(WIKI_API, params={
            "action": "query", "titles": title, "prop": "pageprops", "format": "json"
        }).json()
        pages = r["query"]["pages"]
        wikidata_id = list(pages.values())[0]["pageprops"].get("wikibase_item")

        if not wikidata_id:
            return False

        # Get Wikidata entity type
        wd = requests.get(WIKIDATA_API.format(wikidata_id)).json()
        entity = wd["entities"][wikidata_id]
        claims = entity.get("claims", {})
        if "P31" in claims:  # instance of
            for inst in claims["P31"]:
                val = inst["mainsnak"]["datavalue"]["value"]["id"]
                if val == "Q5":  # Human
                    return True
        return False
    except Exception:
        return False


def harvest_wikipedia_person(name):
    """Get bio, outgoing links, and incoming mentions for a person"""
    people, relations, mentions = [], [], []

    # Get page summary
    summary = requests.get(WIKI_API, params={
        "action": "query", "format": "json", "prop": "extracts",
        "titles": name, "exintro": True, "explaintext": True
    }).json()

    pages = summary["query"]["pages"]
    page = list(pages.values())[0]
    if "missing" in page:
        st.error(f"❌ No page found for {name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Add to people.csv
    people.append({"name": page["title"], "summary": page.get("extract", "")})

    # Get outgoing links
    out_links = requests.get(WIKI_API, params={
        "action": "query", "format": "json", "titles": name,
        "prop": "links", "pllimit": "max"
    }).json()

    links = list(out_links["query"]["pages"].values())[0].get("links", [])
    for l in links:
        if l["ns"] == 0 and is_human(l["title"]):  # only humans
            relations.append({"source": name, "target": l["title"]})

    # Get incoming links (mentions)
    in_links = requests.get(WIKI_API, params={
        "action": "query", "format": "json", "list": "backlinks",
        "bltitle": name, "bllimit": "max"
    }).json()

    backlinks = in_links["query"]["backlinks"]
    for b in backlinks:
        if b["ns"] == 0 and is_human(b["title"]):  # only humans
            mentions.append({"source": b["title"], "target": name})

    return pd.DataFrame(people), pd.DataFrame(relations), pd.DataFrame(mentions)


def build_graph(relations, mentions):
    """Build a graph from relations + mentions"""
    G = nx.Graph()
    for _, row in relations.iterrows():
        G.add_edge(row["source"], row["target"])
    for _, row in mentions.iterrows():
        G.add_edge(row["source"], row["target"])
    return G


# ============ UI ============
st.sidebar.header("🔍 Search or Upload Data")
mode = st.sidebar.radio("Choose mode:", ["Search Wikipedia", "Upload CSV"])

if mode == "Search Wikipedia":
    person_name = st.text_input("Enter a person's name", "Gabriela Mistral")

    if st.button("Harvest Data"):
        p_df, r_df, m_df = harvest_wikipedia_person(person_name)

        if not p_df.empty:
            st.success("✅ Data harvested successfully!")

            # Show CSVs
            st.subheader("👤 People")
            st.dataframe(p_df)

            st.subheader("🔗 Relations (Outgoing)")
            st.dataframe(r_df)

            st.subheader("📥 Mentions (Incoming)")
            st.dataframe(m_df)

            # Save CSVs
            p_df.to_csv("people.csv", index=False)
            r_df.to_csv("relations.csv", index=False)
            m_df.to_csv("mentions.csv", index=False)

            st.download_button("💾 Download people.csv", p_df.to_csv(index=False), "people.csv")
            st.download_button("💾 Download relations.csv", r_df.to_csv(index=False), "relations.csv")
            st.download_button("💾 Download mentions.csv", m_df.to_csv(index=False), "mentions.csv")

            # Build graph
            G = build_graph(r_df, m_df)

            st.subheader("🌐 Network Graph")
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)
            net.show("graph.html")
            st.components.v1.html(open("graph.html").read(), height=600)

            # Graph analytics
            st.subheader("📊 Graph Analytics")
            degree_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            stats = pd.DataFrame({
                "Node": list(degree_centrality.keys()),
                "Degree Centrality": list(degree_centrality.values()),
                "Betweenness": list(betweenness.values())
            })
            st.dataframe(stats)

elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"], accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            df = pd.read_csv(file)
            st.subheader(f"📂 {file.name}")
            st.dataframe(df.head())
