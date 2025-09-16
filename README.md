# 🧠 Great Debate

Great Debate is a research tool for exploring **scholarly perspectives on AI sentience**.
It harvests papers from multiple sources, classifies their stance (Yes / No / Uncertain), and presents interactive visualizations.

---

## ✨ Features

* **Harvest scholarly articles** from arXiv, OpenAlex, and CrossRef.
* **Classify stance** on AI sentience using HuggingFace zero-shot classification with regex safeguards.
* **Store results** in a lightweight SQLite database.
* **Interactive dashboard** (Streamlit) with filters, charts, and representative examples.

---

## 📂 Project Structure

```
GreatDebate/
├── scripts/
│   └── run_harvest.py     # Collect and analyze articles
├── src/ai_opinion/        # Core package (sources, NLP, DB, pipeline)
├── app/
│   └── streamlit_app.py   # Dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Barthmalemew/GreatDebate.git
cd GreatDebate
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Harvest and analyze

```bash
PYTHONPATH=src python scripts/run_harvest.py --report
```

Options:

* `--query "custom terms"` – override default queries
* `--start-year 2020` – fetch from a given year
* `--max-records 500` – cap results
* `--report` – show summary

### 2. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## 📊 Example Outputs

* Distribution of stances (Yes / No / Uncertain)
* Trends in opinion over time
* Representative article snippets with links
* Keyword themes from abstracts

---

## 🧩 Tech Stack

* **Python** (3.11+)
* **NLP**: HuggingFace Transformers, regex patterns
* **Visualization**: Streamlit, Altair, Pandas
* **Database**: SQLite
* **Data Sources**: arXiv, OpenAlex, CrossRef

---

## 👤 Author

**Kevin Anderson**

* 🌐 [Portfolio](https://Kevin-J-Anderson.com)
* 💼 [LinkedIn](https://www.linkedin.com/in/kevinrouse/)
* 💻 [GitHub](https://github.com/Barthmalemew)

---

## 📜 License

MIT License


