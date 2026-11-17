# MarketPulse – social-listening marketing intelligence for small eco brands

Web Data Mining mini-project (Shaunak Wanelkar, VJTI). A small D2C brand selling on Amazon, Blinkit and a
sustainable marketplace enters its keywords, posts, inventory and (optionally) reviews and sales. MarketPulse finds
trending themes and sentiment, keeps only what the brand can fulfil from stock, and writes campaign ideas.

## Run it on your computer
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the link it prints (usually http://localhost:8501), keep **Sample data** selected and click **🚀 Run analysis**.

## Put it online (Streamlit Community Cloud, free)
1. Create a GitHub repository and upload everything in this folder (`app.py`, `pipeline.py`, `sample_data.py`,
   `requirements.txt`, `.streamlit/config.toml`, `README.md`).
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**.
3. Choose your repository, branch `main`, main file `app.py`, then **Deploy**. You get a public `…streamlit.app` link.
4. Optional keys (never commit them to GitHub): in the app's **Settings → Secrets** add
   ```toml
   X_BEARER_TOKEN = "your X API token"
   ANTHROPIC_API_KEY = "your Anthropic key"
   ```
   The app pre-fills these; users can also paste keys into the password boxes.

## Inputs (CSV or Excel; templates downloadable in the app)
| Input | Required columns | Optional |
|---|---|---|
| Posts | `text`, `created_at` | `post_id`, `likes` |
| Inventory | `sku`, `product`, `stock_units` | `category`, `channels`, `tags` |
| Reviews | `sku`, `date`, `stars`, `text` | `review_id` |
| Weekly sales | `sku`, `week_start`, `units` | |
| Labelled posts | `text`, `label` (positive/neutral/negative) | |

## Pipeline (pipeline.py)
Cleaning → MinHash/LSH near-duplicate removal → stop words + Porter stemming → TF-IDF + cosine relevance with query
expansion → rising terms → NMF (default) or LDA themes chosen by NPMI coherence → sentiment (VADER vs logistic
regression) → reviews: fake-review clusters, estimated NPS, aspect complaints, greenwashing rule → theme table →
stock filter → sales correlation (estimate) → campaign copy (template or LLM grounded in mined facts) → evaluation.

## Data and ethics
No scraping. Posts come from the official X API (pay-per-use), the user's own uploads, or the built-in **synthetic**
sample (fictional brand "EcoNest"). Only text, date and likes are kept — no usernames or profiles.

## Tests
`cd tests && python test_app.py` runs the app headlessly with Streamlit's AppTest.
