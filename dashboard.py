# dashboard.py
import io
import os
import re
import json
import time
import tempfile
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd
import networkx as nx

# HTTP / APIs
import requests
import wikipediaapi
from pyvis.network import Network

# Optional: Wikidata enrichment
try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    SPARQL_OK = True
except Exception:
    SPARQL_OK = False

# ----------------------------
# ---------- Helpers ----------
# ----------------------------

def _safe_request_json(url: str, params: dict, headers: dict | None = None, max_retries: int = 3, pause: float = 2.0):
    """
    Safely GET JSON with retries. Returns dict or None.
    """
    headers = headers or {}
    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            # Wikipedia occasionally returns text/html on errors (rate limit, etc.)
            ctype = r.headers.get("Content-Type", "")
            if "application/json" not in ctype and "json" not in ctype:
                last_err = f"Non-JSON response ({ctype}) from {r.url}"
                time.sleep(pause)
                continue
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = str(e)
            time.sleep(pause)
    # Log once to Streamlit (non-fatal)
    st.warning(f"⚠️ API JSON fetch failed: {last_err}")
    return None


def _is_probable_person(name: str) -> bool:
    """
    Simple heuristic to keep human names (Title Case, at least 2 tokens).
    Doesn't enforce—enabled via UI toggle.
    """
    if not isinstance(name, str):
        return False
    name = name.strip()
    return bool(re.match(r'^[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\.\'-]+(?:\s+[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ\.\'-]+)+$', name))


@st.cache_data(show_spinner=False)
def harvest_wikipedia_person(name: str, user_agent: str = "GMNetworkApp/1.0") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Harvest a single person's Wikipedia page:
      - outgoing links (page.links via wikipedia-api)
      - incoming links (MediaWiki backlinks API)
    Returns:
      people_df: rows of unique names (seed + discovered)
      relations_df: edges [source, target, direction ∈ {"out","in"}]
      metadata_df: placeholder; can be enriched with Wikidata
    Robust to API hiccups (safe JSON, retries).
    """
    # Set up Wikipedia client with explicit UA
    wiki = wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent
    )
    page = wiki.page(name)

    # Basic checks
    if not page.exists():
        st.warning(f"'{name}' page does not exist on English Wikipedia.")
        return (pd.DataFrame(columns=["name"]),
                pd.DataFrame(columns=["source", "target", "direction"]),
                pd.DataFrame(columns=["name","birth_date","death_date","description","occupation","nationality","gender"]))

    if "may refer to" in page.summary[:120].lower():
        st.warning(f"'{name}' looks like a disambiguation page. Please verify manually.")

    # --- Outgoing links (from the page) ---
    outgoing_titles = list(page.links.keys())

    # --- Incoming links (backlinks API) ---
    base = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "backlinks",
        "bltitle": name,
        "bllimit": 500,
        "blnamespace": 0,          # main/article namespace only
        "format": "json",
        "redirects": 1
    }
    headers = {"User-Agent": user_agent}
    backlinks_titles = []

    # Paginate safely
    while True:
        data = _safe_request_json(base, params, headers=headers, max_retries=3, pause=2.5)
        if not data:
            break  # don't crash; return what we have

        links = data.get("query", {}).get("backlinks", [])
        backlinks_titles.extend([l.get("title") for l in links if "title" in l])

        if "continue" in data:
            params.update(data["continue"])
            # Gentle pacing to respect rate limits
            time.sleep(0.3)
        else:
            break

    # Build edges
    out_df = pd.DataFrame({
        "source": name,
        "target": outgoing_titles,
        "direction": "out"
    })
    in_df = pd.DataFrame({
        "source": backlinks_titles,
        "target": name,
        "direction": "in"
    })

    # People list includes the seed
    people = set(out_df["target"].dropna().tolist()) | set(in_df["source"].dropna().tolist()) | {name}
    people_df = pd.DataFrame(sorted(people), columns=["name"])

    relations_df = pd.concat([out_df, in_df], ignore_index=True).dropna()

    # Placeholder metadata (optional enrichment later)
    metadata_df = pd.DataFrame(columns=[
        "name","birth_date","death_date","description","occupation","nationality","gender"
    ])

    return people_df, relations_df, metadata_df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip().lower().replace(" ", "_") for c in df2.columns]
    return df2


def apply_basic_cleaning(df: pd.DataFrame,
                         drop_dupes: bool,
                         missing_action: str,
                         trim_text: bool,
                         add_exclude: bool) -> pd.DataFrame:
    d = df.copy()
    if drop_dupes:
        d = d.drop_duplicates()

    if missing_action == "Drop rows":
        d = d.dropna()
    elif missing_action == "Fill with placeholder":
        d = d.fillna("MISSING")

    if trim_text:
        for col in d.select_dtypes(include="object").columns:
            d[col] = d[col].astype(str).str.strip()

    if add_exclude and "exclude" not in d.columns:
        d["exclude"] = False

    return d


def to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_graph_from_edges(edges: pd.DataFrame,
                           source_col: str,
                           target_col: str,
                           directed: bool) -> nx.Graph:
    edges = edges.dropna(subset=[source_col, target_col])
    if directed:
        G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col, create_using=nx.DiGraph)
    else:
        G = nx.from_pandas_edgelist(edges, source=source_col, target=target_col)
    return G


def ego_n_hops(G: nx.Graph, node: str, n: int) -> nx.Graph:
    if node not in G:
        return nx.DiGraph() if G.is_directed() else nx.Graph()
    nodes = {node}
    frontier = {node}
    for _ in range(n):
        nxt = set()
        if G.is_directed():
            for u in frontier:
                nxt |= set(G.predecessors(u))
                nxt |= set(G.successors(u))
        else:
            for u in frontier:
                nxt |= set(G.neighbors(u))
        nodes |= nxt
        frontier = nxt
    return G.subgraph(nodes).copy()


def add_edge_reciprocity_attributes(G: nx.DiGraph) -> None:
    if not G.is_directed():
        return
    for u, v in G.edges():
        G[u][v]["reciprocal"] = G.has_edge(v, u)


def compute_graph_metrics(G: nx.Graph) -> Dict[str, pd.DataFrame]:
    metrics = {}
    deg = dict(G.degree())
    metrics["degree"] = pd.DataFrame({"node": list(deg.keys()), "degree": list(deg.values())}).sort_values("degree", ascending=False)

    try:
        metrics["betweenness"] = pd.DataFrame({"node": list(G.nodes()), "betweenness": list(nx.betweenness_centrality(G).values())}).sort_values("betweenness", ascending=False)
    except Exception:
        metrics["betweenness"] = pd.DataFrame(columns=["node","betweenness"])

    try:
        metrics["closeness"] = pd.DataFrame({"node": list(G.nodes()), "closeness": list(nx.closeness_centrality(G).values())}).sort_values("closeness", ascending=False)
    except Exception:
        metrics["closeness"] = pd.DataFrame(columns=["node","closeness"])

    try:
        metrics["eigenvector"] = pd.DataFrame({"node": list(G.nodes()), "eigenvector": list(nx.eigenvector_centrality(G, max_iter=1000).values())}).sort_values("eigenvector", ascending=False)
    except Exception:
        metrics["eigenvector"] = pd.DataFrame(columns=["node","eigenvector"])

    if G.is_directed():
        try:
            pr = nx.pagerank(G)
            metrics["pagerank"] = pd.DataFrame({"node": list(pr.keys()), "pagerank": list(pr.values())}).sort_values("pagerank", ascending=False)
        except Exception:
            metrics["pagerank"] = pd.DataFrame(columns=["node","pagerank"])
        try:
            global_recip = nx.reciprocity(G)
        except Exception:
            global_recip = None
    else:
        metrics["pagerank"] = pd.DataFrame(columns=["node","pagerank"])
        global_recip = None

    try:
        if G.is_directed():
            comps = list(nx.weakly_connected_components(G))
            largest = G.subgraph(max(comps, key=len)).copy() if comps else G
            comp_count = len(comps)
        else:
            comps = list(nx.connected_components(G))
            largest = G.subgraph(max(comps, key=len)).copy() if comps else G
            comp_count = len(comps)
        density = nx.density(G)
        try:
            diameter = nx.diameter(largest.to_undirected() if G.is_directed() else largest)
        except Exception:
            diameter = None
    except Exception:
        comp_count, density, diameter = None, None, None

    # Communities (greedy modularity on undirected projection)
    try:
        UG = G.to_undirected() if G.is_directed() else G
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(UG))
        memberships = []
        for i, cset in enumerate(comms):
            for n in cset:
                memberships.append((n, i))
        metrics["communities"] = pd.DataFrame(memberships, columns=["node","community"])
        metrics["community_count"] = pd.DataFrame([{"communities": len(comms)}])
    except Exception:
        metrics["communities"] = pd.DataFrame(columns=["node","community"])
        metrics["community_count"] = pd.DataFrame([{"communities": None}])

    metrics["summary"] = pd.DataFrame([{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "components": comp_count,
        "density": density,
        "diameter_largest_component": diameter,
        "global_reciprocity": global_recip
    }])

    return metrics


def export_graph_files(G: nx.Graph) -> Dict[str, bytes]:
    out = {}
    edges_df = nx.to_pandas_edgelist(G)
    out["edges.csv"] = edges_df.to_csv(index=False).encode("utf-8")
    nodes_df = pd.DataFrame({"node": list(G.nodes())})
    out["nodes.csv"] = nodes_df.to_csv(index=False).encode("utf-8")
    with io.BytesIO() as buf:
        nx.write_gexf(G, buf)
        out["graph.gexf"] = buf.getvalue()
    with io.BytesIO() as buf:
        nx.write_graphml(G, buf)
        out["graph.graphml"] = buf.getvalue()
    return out


def pyvis_html(G: nx.Graph, height: str = "650px", notebook: bool = False) -> str:
    net = Network(height=height, width="100%", notebook=notebook, bgcolor="#111111", font_color="white")
    degrees = dict(G.degree())
    for n in G.nodes():
        size = 10 + 2 * degrees.get(n, 1)
        net.add_node(str(n), label=str(n), value=size)
    for u, v, data in G.edges(data=True):
        title = ", ".join([f"{k}: {v}" for k, v in data.items() if v is not None])
        net.add_edge(str(u), str(v), title=title if title else None)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        # write_html avoids trying to open a browser
        net.write_html(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
    return html


def enrich_people_with_wikidata(names: List[str]) -> pd.DataFrame:
    cols = ["name","birth_date","death_date","description","occupation","nationality","gender"]
    if not SPARQL_OK or not names:
        return pd.DataFrame(columns=cols)

    all_rows = []
    endpoint = SPARQLWrapper("https://query.wikidata.org/sparql")
    for i in range(0, len(names), 15):
        batch = names[i:i+15]
        filter_names = " ".join(f'"{n}"@en' for n in batch)
        q = f"""
        SELECT ?personLabel ?birth_date ?death_date ?description ?occupationLabel ?nationalityLabel ?genderLabel WHERE {{
          VALUES ?personLabel {{ {filter_names} }}
          ?person rdfs:label ?personLabel.
          OPTIONAL {{ ?person wdt:P569 ?birth_date. }}
          OPTIONAL {{ ?person wdt:P570 ?death_date. }}
          OPTIONAL {{ ?person schema:description ?description. FILTER (lang(?description) = "en") }}
          OPTIONAL {{ ?person wdt:P106 ?occupation. }}
          OPTIONAL {{ ?person wdt:P27 ?nationality. }}
          OPTIONAL {{ ?person wdt:P21 ?gender. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        endpoint.setQuery(q)
        endpoint.setReturnFormat(JSON)
        try:
            results = endpoint.query().convert()
            for b in results["results"]["bindings"]:
                all_rows.append({
                    "name": b.get("personLabel", {}).get("value"),
                    "birth_date": b.get("birth_date", {}).get("value"),
                    "death_date": b.get("death_date", {}).get("value"),
                    "description": b.get("description", {}).get("value"),
                    "occupation": b.get("occupationLabel", {}).get("value"),
                    "nationality": b.get("nationalityLabel", {}).get("value"),
                    "gender": b.get("genderLabel", {}).get("value"),
                })
        except Exception:
            time.sleep(5)
            continue

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["name"])
    return df


# ---------------------------------
# ----------- UI Layout -----------
# ---------------------------------

st.set_page_config(page_title="Data Harvesting & Network Dashboard", layout="wide")
st.title("📚 Data Harvesting & Network Dashboard")

tabs = st.tabs([
    "1) Harvest & Clean",
    "2) Graph Builder",
    "3) Analytics",
    "4) Exports",
    "5) Limitations"
])

# ----------------------------
# 1) Harvest & Clean
# ----------------------------
with tabs[0]:
    st.subheader("Harvest Wikipedia → Incoming/Outgoing Links")
    names_text = st.text_area(
        "Enter one or more names (comma-separated). Example: Gabriela Mistral, Pablo Neruda",
        value="Gabriela Mistral"
    )
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        do_enrich = st.checkbox("Enrich with Wikidata (birth/death/description/occupation/nationality/gender)")
    with c2:
        normalize_names = st.checkbox("Normalize harvested names (trim whitespace)", value=True)
    with c3:
        only_probable_people = st.checkbox("Keep only probable person names in links", value=True)

    if st.button("Harvest Now"):
        seeds = [n.strip() for n in names_text.split(",") if n.strip()]
        all_people_frames, all_relations_frames, all_meta_frames = [], [], []

        for nm in seeds:
            p_df, r_df, m_df = harvest_wikipedia_person(nm)

            # Optional filter to keep only likely persons in relations
            if only_probable_people and not r_df.empty:
                # For OUT edges: filter target; for IN edges: filter source
                mask = []
                for _, row in r_df.iterrows():
                    if row.get("direction") == "out":
                        mask.append(_is_probable_person(row.get("target", "")))
                    else:
                        mask.append(_is_probable_person(row.get("source", "")))
                r_df = r_df[mask]

            # Normalize columns
            p_df = normalize_columns(p_df)
            r_df = normalize_columns(r_df)
            m_df = normalize_columns(m_df)

            if normalize_names and "name" in p_df.columns:
                p_df["name"] = p_df["name"].astype(str).str.strip()

            all_people_frames.append(p_df)
            all_relations_frames.append(r_df)
            all_meta_frames.append(m_df)

        persons_raw = pd.concat(all_people_frames, ignore_index=True).drop_duplicates() if all_people_frames else pd.DataFrame(columns=["name"])
        relations_raw = pd.concat(all_relations_frames, ignore_index=True).drop_duplicates() if all_relations_frames else pd.DataFrame(columns=["source","target","direction"])
        metadata_raw  = pd.concat(all_meta_frames, ignore_index=True).drop_duplicates() if all_meta_frames else pd.DataFrame(columns=["name","birth_date","death_date","description","occupation","nationality","gender"])

        # Optional enrichment
        if do_enrich:
            current_names = set(persons_raw["name"].dropna().unique().tolist())
            # also include nodes from relations
            current_names |= set(relations_raw["source"].dropna().unique().tolist())
            current_names |= set(relations_raw["target"].dropna().unique().tolist())
            enriched = enrich_people_with_wikidata(sorted(current_names))
            if not enriched.empty:
                metadata_raw = pd.concat([metadata_raw, enriched], ignore_index=True)
                metadata_raw = metadata_raw.drop_duplicates(subset=["name"], keep="last")

        st.success("✅ Harvest complete")
        st.write("**Persons (raw)**", persons_raw.head(20))
        st.write("**Relations (raw)**", relations_raw.head(20))
        st.write("**Metadata (raw)**", metadata_raw.head(20))

        st.session_state["persons_raw"] = persons_raw
        st.session_state["relations_raw"] = relations_raw
        st.session_state["metadata_raw"] = metadata_raw

    st.markdown("---")
    st.subheader("Interactive Cleaning")

    left, right = st.columns(2)
    with left:
        up_persons = st.file_uploader("Upload persons CSV (optional)", type=["csv"], key="up_p")
    with right:
        up_relations = st.file_uploader("Upload relations CSV (optional)", type=["csv"], key="up_r")

    if up_persons is not None:
        persons_src = normalize_columns(pd.read_csv(up_persons))
    else:
        persons_src = st.session_state.get("persons_raw", pd.DataFrame(columns=["name"]))

    if up_relations is not None:
        relations_src = normalize_columns(pd.read_csv(up_relations))
    else:
        relations_src = st.session_state.get("relations_raw", pd.DataFrame(columns=["source","target","direction"]))

    metadata_src = st.session_state.get("metadata_raw",
                                        pd.DataFrame(columns=["name","birth_date","death_date","description","occupation","nationality","gender"]))

    st.write("**Preview – Persons (source)**", persons_src.head())
    st.write("**Preview – Relations (source)**", relations_src.head())
    st.write("**Preview – Metadata (source)**", metadata_src.head())

    st.markdown("**Cleaning Options**")
    c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.2])
    with c1:
        drop_dupes = st.checkbox("Remove duplicates", value=True)
    with c2:
        missing_action = st.radio("Missing values", ["Do nothing", "Drop rows", "Fill with placeholder"], index=0, horizontal=True)
    with c3:
        trim_text = st.checkbox("Trim text cells", value=True)
    with c4:
        add_exclude = st.checkbox("Add 'exclude' column", value=False)

    persons_clean = apply_basic_cleaning(persons_src, drop_dupes, missing_action, trim_text, add_exclude)
    relations_clean = apply_basic_cleaning(relations_src, drop_dupes, missing_action, trim_text, add_exclude)
    metadata_clean = apply_basic_cleaning(metadata_src, drop_dupes, missing_action, trim_text, add_exclude)

    # Ensure 'source'/'target' exist if user uploaded different headers
    if "source" not in relations_clean.columns or "target" not in relations_clean.columns:
        if len(relations_clean.columns) >= 2:
            guess_cols = relations_clean.columns[:2].tolist()
            relations_clean = relations_clean.rename(columns={guess_cols[0]: "source", guess_cols[1]: "target"})
        else:
            st.warning("Relations file requires at least two columns (source/target).")

    st.success("✅ Cleaning applied")
    st.write("**Persons (clean)**", persons_clean.head())
    st.write("**Relations (clean)**", relations_clean.head())
    st.write("**Metadata (clean)**", metadata_clean.head())

    st.session_state["persons_clean"] = persons_clean
    st.session_state["relations_clean"] = relations_clean
    st.session_state["metadata_clean"] = metadata_clean

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("⬇️ Download persons_clean.csv", to_bytes_csv(persons_clean), file_name="persons_clean.csv")
    with d2:
        st.download_button("⬇️ Download relations_clean.csv", to_bytes_csv(relations_clean), file_name="relations_clean.csv")
    with d3:
        st.download_button("⬇️ Download metadata_clean.csv", to_bytes_csv(metadata_clean), file_name="metadata_clean.csv")


# ----------------------------
# 2) Graph Builder
# ----------------------------
with tabs[1]:
    st.subheader("Create Graph from Cleaned Data")
    persons_clean = st.session_state.get("persons_clean", pd.DataFrame(columns=["name"]))
    relations_clean = st.session_state.get("relations_clean", pd.DataFrame(columns=["source","target","direction"]))
    st.write("**Relations available**", relations_clean.head())

    if relations_clean.empty or relations_clean.shape[1] < 2:
        st.warning("Please produce/attach a valid relations_clean.csv with at least two columns.")
        st.stop()

    cols = relations_clean.columns.tolist()
    source_col = st.selectbox("Source column", cols, index=cols.index("source") if "source" in cols else 0)
    target_col = st.selectbox("Target column", cols, index=cols.index("target") if "target" in cols else min(1, len(cols)-1))
    directed = st.checkbox("Directed graph", value=True)
    min_degree = st.slider("Minimum degree filter (on final graph)", 0, 50, 0)

    G_full = build_graph_from_edges(relations_clean, source_col, target_col, directed)
    if directed:
        add_edge_reciprocity_attributes(G_full)

    st.markdown("**n-Hops Mesh / Ego Network**")
    nodes_list = sorted(list(G_full.nodes()))
    if nodes_list:
        default_idx = nodes_list.index("Gabriela Mistral") if "Gabriela Mistral" in nodes_list else 0
        seed = st.selectbox("Pick a person for ego network", nodes_list, index=default_idx)
        hops = st.slider("Number of hops (n)", 0, 4, 1)
        G_view = ego_n_hops(G_full, seed, hops)
    else:
        seed = None
        G_view = G_full

    degs = dict(G_view.degree())
    keep_nodes = [n for n, d in degs.items() if d >= min_degree]
    G_view = G_view.subgraph(keep_nodes).copy()

    st.write(f"Graph view: **{G_view.number_of_nodes()} nodes** / **{G_view.number_of_edges()} edges**")

    html = pyvis_html(G_view)
    st.components.v1.html(html, height=680, scrolling=True)

    st.session_state["G_view"] = G_view
    st.session_state["G_full"] = G_full


# ----------------------------
# 3) Analytics
# ----------------------------
with tabs[2]:
    st.subheader("Graph Analytics & Indices")
    G_view = st.session_state.get("G_view", None)
    if G_view is None or G_view.number_of_nodes() == 0:
        st.warning("Build a graph in the previous tab first.")
        st.stop()

    metrics = compute_graph_metrics(G_view)
    st.write("**Summary**")
    st.dataframe(metrics["summary"])

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Degree**")
        st.dataframe(metrics["degree"].head(20))
        st.write("**Betweenness**")
        st.dataframe(metrics["betweenness"].head(20))
    with c2:
        st.write("**Closeness**")
        st.dataframe(metrics["closeness"].head(20))
        st.write("**Eigenvector**")
        st.dataframe(metrics["eigenvector"].head(20))

    if not metrics["pagerank"].empty:
        st.write("**PageRank**")
        st.dataframe(metrics["pagerank"].head(20))

    if not metrics["communities"].empty:
        st.write("**Communities (greedy modularity, undirected projection)**")
        st.dataframe(metrics["communities"].sort_values("community").head(50))

    st.caption("Reciprocity index (global) is shown in Summary (directed graphs only). Each directed edge is annotated with 'reciprocal=True/False' in the visualization.")


# ----------------------------
# 4) Exports
# ----------------------------
with tabs[3]:
    st.subheader("Download Data & Graph Files")
    G_full = st.session_state.get("G_full", None)
    G_view = st.session_state.get("G_view", None)

    persons_clean = st.session_state.get("persons_clean", pd.DataFrame())
    relations_clean = st.session_state.get("relations_clean", pd.DataFrame())
    metadata_clean = st.session_state.get("metadata_clean", pd.DataFrame())

    st.write("**Cleaned CSVs**")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("persons_clean.csv", to_bytes_csv(persons_clean), file_name="persons_clean.csv")
    with e2:
        st.download_button("relations_clean.csv", to_bytes_csv(relations_clean), file_name="relations_clean.csv")
    with e3:
        st.download_button("metadata_clean.csv", to_bytes_csv(metadata_clean), file_name="metadata_clean.csv")

    st.markdown("---")
    st.write("**Graph Files (current view)**")
    if G_view is not None and G_view.number_of_nodes() > 0:
        files = export_graph_files(G_view)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("nodes.csv", files["nodes.csv"], file_name="nodes.csv")
        with c2:
            st.download_button("edges.csv", files["edges.csv"], file_name="edges.csv")
        with c3:
            st.download_button("graph.gexf (Gephi)", files["graph.gexf"], file_name="graph.gexf")
        with c4:
            st.download_button("graph.graphml", files["graph.graphml"], file_name="graph.graphml")
    else:
        st.info("No graph view available. Build a graph first in tab 2.")

    st.caption("Tip: Use the GEXF in Gephi for adapted mesh visualisation and further styling.")


# ----------------------------
# 5) Limitations & Assumptions
# ----------------------------
with tabs[4]:
    st.subheader("Limitations & Assumptions")
    st.markdown("""
- We **assume** the selected Wikipedia page(s) are correct for each person (disambiguation requires manual verification).
- Wikipedia/Wikidata may be **incomplete or inconsistent**, so harvested links and metadata can be noisy.
- MediaWiki API and Wikidata SPARQL endpoints are **rate-limited**; enrichment may skip/slow when throttled.
- The **reciprocity index** and community detection depend on graph direction and density; interpret with care.
- Centrality measures on **disconnected or tiny graphs** can be unstable or uninformative.
- The PyVis rendering is **browser-based**; very large graphs may render slowly. Use Gephi for heavy layouts.
- Reproducibility: ensure consistent seeds & snapshots of data if you publish figures.
""")

    st.markdown("**Reproducibility Checklist (GitHub):**")
    st.markdown("""
- ✅ `dashboard.py` (this app)
- ✅ `requirements.txt`
- ✅ Example data: `persons_clean.csv`, `relations_clean.csv`, `metadata_clean.csv`
- ✅ README with steps to **collect → clean → build → analyze → visualize**
- ✅ Screenshots / short notes of experiments (e.g., list of intellectuals, scaling the study)
""")
