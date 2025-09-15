# AI-Powered EDA Assistant

A smart, interactive data analysis assistant built using Chainlit, Pandas, Seaborn, Matplotlib, and Google Gemini AI.
This app allows users to upload a CSV dataset, automatically generates insights, visualizations, and AI-driven analysis plans, all in an interactive chat interface.

---

## Features

  - CSV Upload: Easily upload any dataset in CSV format.
  - Data Summary: Quick overview of schema, missing values, and sample rows.
  - Interactive Visualizations: Automatically generates:
    - Correlation Heatmaps
    - Pairplots
    - Histograms, Boxplots, and Violin plots
    - Countplots for categorical features
  - Sidebar Graph View: Click on a visualization title to open it in a resizable sidebar.
  - AI Insights: Powered by Google Gemini AI:
    - Suggests a concise analysis plan
    - Provides final insights and summaries of your data
  - Dark-Themed Advanced UI: Sleek black background with green & white text for better readability.
  - Lightweight & Fast: All computations run locally with temporary graph storage.

---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/Samima-Nasrin/EDA-agent.git
cd EDA-agent
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
# Create a .env file in the root directory:
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=your_gemini_model_name
```

---

## Usage

Run the app:
```bash
chainlit run app.py
```
Open your browser at [http://localhost:8000]
and start interacting with the AI assistant.
  - Upload your CSV dataset
  - Review AI-generated summary
  - Click visualization titles to view graphs in sidebar
  - Explore AI insights

---

## Author
Made by Samima Nasrin.
