# Interactive Data Harvesting & Network Dashboard

This project provides an **interactive dashboard** for harvesting, cleaning, and analyzing Wikipedia-based biographical data, focusing on Gabriela Mistral and her intellectual network.  
It integrates **data collection, cleaning, graph building, and analytics** into a single Streamlit web app.

---

## ✨ Features

### 1. Data Harvesting
- **What it does**: Collects metadata and relations for a given person (e.g., Gabriela Mistral) directly from Wikipedia. It captures:
  - Outgoing links (people linked from the biography)
  - Incoming links (people linking back to the biography)
  - Mentions across related articles
- **Why it matters**: Instead of manually browsing Wikipedia, the app automatically builds datasets of intellectual networks.
- **How to use**: In the sidebar, enter a person’s name (e.g., *Gabriela Mistral*) and click **Harvest Data**. The app will fetch related people, relations, and mentions, then save them as CSV files.

---

### 2. Data Cleaning
- **What it does**: Provides an interactive process for refining harvested data.
  - Removes duplicates
  - Handles missing values
  - Normalizes text
  - Adds “Exclude” column for manual filtering
- **Why it matters**: Raw Wikipedia data contains noise (non-person entities, duplicates, missing info). Cleaning ensures high-quality datasets.
- **How to use**: After harvesting/uploading data, select **Cleaning Options** from the sidebar. Adjust checkboxes (e.g., “Remove duplicates”) and download the cleaned CSV.

---

### 3. Building Graphs
- **What it does**: Transforms the cleaned CSVs into **network graphs**.
  - Nodes = People
  - Edges = Relationships (outgoing/incoming links)
- **Why it matters**: Networks make it possible to see intellectual communities and visualize how people are connected.
- **How to use**: Upload your cleaned CSVs (or use harvested data) and select the **Build Graph** tab. The graph will display interactively.

---

### 4. Graph Analytics
- **What it does**: Computes network measures such as:
  - Degree centrality (importance by number of links)
  - Reciprocity (two-way connections)
  - Connected components
- **Why it matters**: Analytics quantify influence and roles within the network.
- **How to use**: Once the graph is generated, switch to the **Analytics** section. The app will show centrality rankings, reciprocity indices, and summary statistics.

---

### 5. Interactive Graph Visualization
- **What it does**: Creates an interactive **PyVis network** you can zoom, pan, and explore.
- **Why it matters**: Large graphs are easier to explore when interactive.
- **How to use**: Go to the **Interactive Graph** tab. Hover over nodes to see metadata (birth date, nationality, occupation). Zoom in/out for exploration.

---

### 6. Wikipedia Person Lookup
- **What it does**: Lets you search for **any person’s name** without uploading a file. The app will:
  - Harvest biography
  - Extract incoming/outgoing links
  - Filter out non-human pages
  - Build graph and analytics automatically
- **Why it matters**: Fast and simple—just type a name, get the full dataset and graph.
- **How to use**: In the sidebar, type a person’s name (e.g., *Pablo Neruda*) and press enter. All related datasets and graphs will be generated instantly.

---

### 7. Export Options
- **What it does**: Allows exporting datasets:
  - `people.csv` — cleaned list of people
  - `relations.csv` — connections between people
  - `mentions.csv` — context mentions in Wikipedia
- **Why it matters**: Data can be reused for further research (e.g., in Gephi or Python notebooks).
- **How to use**: After harvesting/cleaning, click **Download CSV**. Use exported files in Gephi or other analysis tools.

---

## 📊 Limitations
- The app assumes Wikipedia pages are correct and up-to-date.  
- Some intellectuals may be under-represented or missing.  
- Filtering still depends on Wikidata tags — some people might slip through if incorrectly classified.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/interactive-data-harvester.git
   cd interactive-data-harvester
