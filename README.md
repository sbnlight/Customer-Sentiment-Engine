# Customer-Sentiment-Engine

A RAG (Retrieval-Augmented Generation) chatbot that scrapes Trustpilot reviews, analyzes their sentiment, and helps startups identify market opportunities and craft customer responses — powered by ChromaDB and the Qwen LLM.

## Pipeline Overview

```
category_auto_crawler_advanced.py   →  Raw reviews (CSV)
         ↓
sentiment_analysis.py               →  Weighted Sentiment Scores (CSV)
         ↓
build_db.py                         →  ChromaDB vector database
         ↓
rag_chat_chainlit_new.py            →  Chatbot UI (Chainlit)
```

## File Structure

```
├── rag_chat_chainlit_new.py          # MAIN APPLICATION — run this to start the chatbot
├── build_db.py                       # Vectorises processed CSV data into ChromaDB
├── sentiment_analysis.py             # Calculates Weighted Sentiment Score from raw reviews
├── category_auto_crawler_advanced.py # Web crawler to fetch reviews from Trustpilot
├── requirements.txt                  # Python dependencies
├── chroma_db.zip                     # Pre-built vector database (ready to use)
├── data_with_sentiment_score.zip     # Processed CSVs with sentiment scores
└── categories_data_big.zip           # Raw scraped review data
```

## Requirements

Python 3.9 or higher is required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start (Pre-built Database)

The vector database (`chroma_db`) is pre-built and included. You can run the chatbot immediately:

1. Unzip `chroma_db.zip` in the project folder.
2. Install dependencies (see above).
3. Set your Qwen API key as an environment variable:
   ```bash
   export QWEN_API_KEY="your-api-key-here"   # macOS/Linux
   set QWEN_API_KEY=your-api-key-here         # Windows
   ```
4. Run the chatbot:
   ```bash
   chainlit run rag_chat_chainlit_new.py -w
   ```
5. The browser will open automatically at `http://localhost:8000`.

## Full Pipeline (From Scratch)

If you want to collect fresh data and rebuild the database:

**Step 1 — Crawl Trustpilot reviews:**
```bash
python category_auto_crawler_advanced.py
```
Output: `categories_data_big/` folder with raw CSVs.

**Step 2 — Run sentiment analysis:**
```bash
python sentiment_analysis.py
```
Output: `data_with_sentiment_score/` folder with scored CSVs.

**Step 3 — Build the vector database:**
```bash
python build_db.py
```
Output: `chroma_db/` folder (vector store ready for the chatbot).

**Step 4 — Start the chatbot** (see Quick Start above).

## Configuration

The chatbot uses the [Qwen (Tongyi Qianwen)](https://www.alibabacloud.com/product/tongyi-qianwen) API via an OpenAI-compatible endpoint.

Set your API key as an environment variable before running:
```bash
export QWEN_API_KEY="your-api-key-here"
```

To use a different model, edit the `model` field in `rag_chat_chainlit_new.py`:
```python
llm = ChatOpenAI(model="qwen-turbo", ...)
```

## What the Chatbot Can Do

- **Market Research** — "What are the common complaints in the home services industry?"
- **Competitor Analysis** — "Which brand has the best weighted score and why?"
- **Customer Support** — "Draft a polite reply to this negative review."

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Module not found` | Run `pip install -r requirements.txt` |
| Chatbot does not reply | Check your `QWEN_API_KEY` environment variable and internet connection |
| `Database not found` | Unzip `chroma_db.zip` or run `build_db.py` to rebuild it |
