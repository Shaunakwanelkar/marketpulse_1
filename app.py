"""MarketPulse – social-listening marketing intelligence for a small eco brand.

Run locally:   pip install -r requirements.txt && streamlit run app.py
Deploy:        push this folder to GitHub, then create an app on Streamlit Community Cloud with main file app.py.
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline as P
import sample_data as S

st.set_page_config(page_title="MarketPulse · Marketing intelligence", page_icon="📣", layout="wide")

# ---------------------------------------------------------------- chart style (validated palette)
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#c3c2b7"
SENT = {"positive": BLUE, "neutral": GRAY, "negative": RED}
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Arial", size=13), margin=dict(l=10, r=20, t=50, b=40),
              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
DECISION_ICON = {"Promote": "🟢", "Restock before promoting": "📦", "Fix-first alert": "🔴",
                 "Clarify eco claims": "🟠", "Brand content": "📝", "Monitor": "👀"}


@st.cache_data(show_spinner=False)
def load_sample():
    return S.generate_sample()


@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    name = getattr(file, "name", "")
    df = pd.read_excel(file) if name.endswith((".xlsx", ".xls")) else pd.read_csv(file)
    return P.normalise_columns(df)


def secret(name: str) -> str:
    """Read an optional secret (Streamlit Cloud → App settings → Secrets) without failing when none exist."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def source_picker(label, key, kinds=("Sample data", "Upload CSV", "None")):
    return st.radio(label, kinds, horizontal=True, key=key)


def uploaded(kind, key):
    f = st.file_uploader(f"Upload {kind} file (CSV or Excel)", type=["csv", "xlsx", "xls"], key=key)
    if f is None:
        return None
    df = read_csv(f)
    errs = P.validate(df, kind)
    if errs:
        st.error(" · ".join(errs) + f". Required: {', '.join(P.SCHEMAS[kind]['required'])}")
        return None
    st.caption(f"✅ {len(df):,} rows loaded")
    return df


# ---------------------------------------------------------------- header
st.title("📣 MarketPulse")
st.markdown("**Social-listening marketing intelligence for small eco brands.** Enter your keywords and data, "
            "and MarketPulse finds what people are talking about, how they feel, what you can actually fulfil from stock, "
            "and turns it into campaign ideas.")

tabs = st.tabs(["⚙️ Setup & run", "📊 Overview", "📈 Trends", "🧩 Themes", "⭐ Reviews", "💰 Sales link",
                "📣 Campaign studio", "✅ Evaluation", "ℹ️ About & ethics"])
sample = load_sample()

# ================================================================= SETUP
with tabs[0]:
    st.subheader("1 · Brand and campaign")
    c1, c2 = st.columns(2)
    brand = c1.text_input("Brand name", "EcoNest")
    occasion = c2.text_input("Occasion / season", "Diwali")
    keywords = st.text_area(
        "Keywords to listen for — one per line. Optionally add related words after a colon (query expansion).",
        S.SAMPLE_KEYWORDS, height=130,
        help="Example line → eco diwali: diya candle gift festive. Include your brand name with service words "
             "(order, delivery) so complaints are not missed.")

    st.subheader("2 · Social posts")
    post_src = st.radio("Where should posts come from?", ["Sample data (synthetic)", "Upload CSV", "X API (official)"],
                        horizontal=True, key="post_src")
    posts = None
    if post_src.startswith("Sample"):
        posts = sample["posts"]
        st.info(f"Using {len(posts):,} **synthetic** X-style posts (14 days before Diwali, fictional brand). "
                "They include retweets, spam copies and off-topic posts so every step has something to do.")
    elif post_src == "Upload CSV":
        st.caption("Columns: `text`, `created_at` (date) · optional `post_id`, `likes`.")
        posts = uploaded("posts", "up_posts")
    else:
        st.warning("Uses X's **official API** (pay-per-use; roughly $0.005 per post read — check X's pricing page). "
                   "Scraping X is against its terms, so this app never scrapes.")
        token = st.text_input("X API bearer token", value=secret("X_BEARER_TOKEN"), type="password")
        per_kw = st.number_input("Posts per keyword", 10, 1000, 100, step=10)
        n_kw = len(P.parse_keywords(keywords))
        st.caption(f"Estimated reads: {per_kw * n_kw:,} posts ≈ ${per_kw * n_kw * 0.005:,.2f}")
        if st.button("Fetch posts from X") and token:
            with st.spinner("Calling the X API…"):
                try:
                    st.session_state["x_posts"] = P.fetch_x_posts(token, list(P.parse_keywords(keywords)), per_kw)
                except Exception as e:
                    st.error(str(e))
        posts = st.session_state.get("x_posts")
        if posts is not None:
            st.caption(f"✅ {len(posts):,} posts fetched")

    st.subheader("3 · Inventory (edit directly)")
    inv_src = st.radio("Inventory source", ["Sample data", "Upload CSV"], horizontal=True, key="inv_src")
    inv_base = sample["inventory"] if inv_src == "Sample data" else uploaded("inventory", "up_inv")
    inventory = None
    if inv_base is not None:
        st.caption("`tags` = words that link a theme to this product (falls back to the product name).")
        inventory = st.data_editor(inv_base, num_rows="dynamic", width="stretch", key=f"inv_editor_{inv_src}",
                                   column_config={"stock_units": st.column_config.NumberColumn("stock_units", min_value=0, step=1)})

    st.subheader("4 · Customer reviews and sales (optional)")
    c1, c2 = st.columns(2)
    with c1:
        rv_src = source_picker("Reviews", "rv_src")
        reviews = sample["reviews"] if rv_src == "Sample data" else uploaded("reviews", "up_rv") if rv_src == "Upload CSV" else None
        st.caption("Columns: `sku`, `date`, `stars` (1–5), `text`. Use your own store export — don't scrape marketplaces.")
    with c2:
        sl_src = source_picker("Weekly sales", "sl_src")
        sales = sample["sales"] if sl_src == "Sample data" else uploaded("sales", "up_sl") if sl_src == "Upload CSV" else None
        st.caption("Columns: `sku`, `week_start`, `units`. Needs reviews too.")

    st.subheader("5 · Sentiment model")
    lb_src = source_picker("Hand-labelled posts for training (text, label = positive / neutral / negative)", "lb_src")
    labelled = sample["labelled"] if lb_src == "Sample data" else uploaded("labelled", "up_lb") if lb_src == "Upload CSV" else None
    sent_choice = st.selectbox("Model", ["auto (best macro-F1 on labelled posts, else VADER)", "VADER (lexicon, no training)",
                                         "LogReg trained on labelled posts", "LogReg trained on review stars"])

    with st.expander("Advanced settings"):
        c1, c2, c3 = st.columns(3)
        topic_method = c1.selectbox("Topic model", ["NMF (recommended for short posts)", "LDA"],
                                    help="LDA is slower (~30 s) and on short posts can merge themes (e.g. complaints into a "
                                         "product theme). Check the Themes tab if you use it.")
        k_range = c1.slider("Number of themes to try", 3, 15, (6, 10))
        rel_th = c2.slider("Relevance threshold (cosine)", 0.02, 0.40, 0.10, 0.01)
        dup_th = c2.slider("Near-duplicate threshold (Jaccard)", 0.60, 0.95, 0.80, 0.05)
        min_stock = c3.number_input("Minimum stock to promote", 0, 100000, 100, step=10)

    st.subheader("6 · Campaign copy")
    copy_mode = st.radio("Generator", ["Template (no API key needed)", "LLM (Anthropic API key)"], horizontal=True)
    llm_key = ""
    if copy_mode.startswith("LLM"):
        llm_key = st.text_input("Anthropic API key", value=secret("ANTHROPIC_API_KEY"),
                                type="password", help="Only mined facts are sent (no usernames).")

    with st.expander("📄 Download empty CSV templates"):
        cols = st.columns(5)
        for col, kind in zip(cols, ["posts", "inventory", "reviews", "sales", "labelled"]):
            col.download_button(kind, csv_bytes(P.template(kind)), f"template_{kind}.csv", "text/csv", key=f"tpl_{kind}")

    st.divider()
    ready = posts is not None and inventory is not None and len(P.parse_keywords(keywords)) > 0
    if not ready:
        st.info("Add posts, inventory and at least one keyword to run.")
    if st.button("🚀 Run analysis", type="primary", disabled=not ready, width="stretch"):
        settings = dict(brand=brand, occasion=occasion, relevance_threshold=rel_th, dedupe_threshold=dup_th,
                        k_min=k_range[0], k_max=k_range[1], min_stock=min_stock, llm_key=llm_key,
                        topic_method="nmf" if topic_method.startswith("NMF") else "lda",
                        sentiment="auto" if sent_choice.startswith("auto") else sent_choice)
        with st.status("Running the pipeline…", expanded=True) as status:
            try:
                R = P.run_pipeline(posts, inventory, keywords, reviews, sales, labelled, settings, progress=st.write)
                st.session_state["R"] = R
                status.update(label="Done — open the other tabs to see results", state="complete")
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(str(e))

R = st.session_state.get("R")


def need_results():
    st.info("Run the analysis on the **⚙️ Setup & run** tab first.")


def daily_chart(R):
    rel, themes = R["relevant"], R["themes"]
    daily = rel.groupby(["created_at", "topic_label"]).size().reset_index(name="posts")
    hl = {}
    top = themes[themes.decision == "Promote"].head(1).theme.tolist()
    alert = themes[themes.decision == "Fix-first alert"].theme.tolist()[:1]
    for lab, c in zip(top + alert, [BLUE, RED]):
        hl[lab] = c
    fig = go.Figure()
    for lab in rel.topic_label.value_counts().index:
        d = daily[daily.topic_label == lab]
        on = lab in hl
        fig.add_trace(go.Scatter(x=d.created_at, y=d.posts, name=lab, mode="lines+markers" if on else "lines",
                                 line=dict(width=2.5 if on else 1.2, color=hl.get(lab, GRAY)), marker=dict(size=7)))
    title = "Daily posts per theme"
    if alert:
        title += f" — '{alert[0]}' is a growing complaint"
    fig.update_layout(title=title, hovermode="x unified", yaxis_title="Posts per day", height=420, **LAYOUT)
    return fig


# ================================================================= OVERVIEW
with tabs[1]:
    if R is None:
        need_results()
    else:
        th = R["themes"]
        m = st.columns(6)
        m[0].metric("Posts collected", f"{R['raw_posts']:,}")
        m[1].metric("Unique after dedupe", f"{R['unique_posts']:,}")
        m[2].metric("Relevant", f"{len(R['relevant']):,}")
        m[3].metric("Themes found", R["topics"]["k"])
        m[4].metric("Campaigns", len(R["campaigns"]))
        m[5].metric("Alerts", int(th.decision.isin(["Fix-first alert", "Clarify eco claims", "Restock before promoting"]).sum()))
        st.plotly_chart(daily_chart(R), width="stretch", key="chart_1")
        st.markdown("#### What to do next")
        for r in th.itertuples():
            prod = f" · {r.product} ({int(r.stock_units)} in stock)" if isinstance(r.product, str) else ""
            line = (f"{DECISION_ICON.get(r.decision, '')} **{r.decision}** — *{r.theme}*: {r.posts} posts, "
                    f"{r.positive:.0%} positive, {r.negative:.0%} negative, growth {r.growth}×{prod}")
            if r.decision == "Fix-first alert":
                st.error(line)
            elif r.decision in ("Clarify eco claims", "Restock before promoting"):
                st.warning(line)
            else:
                st.markdown(line)
        if R.get("llm_error"):
            st.caption(f"LLM copy failed, template copy used instead: {R['llm_error'][:150]}")

# ================================================================= TRENDS
with tabs[2]:
    if R is None:
        need_results()
    else:
        rising = R["rising"].head(15)
        c1, c2 = st.columns([3, 2])
        if len(rising):
            r = rising.iloc[::-1]
            fig = go.Figure(go.Bar(x=r.growth_ratio, y=r.term, orientation="h", marker_color=BLUE,
                                   customdata=np.stack([r.mentions_recent, r.mentions_before], axis=1),
                                   hovertemplate="%{y}: %{x}× <br>last 3 days %{customdata[0]} · before %{customdata[1]}<extra></extra>"))
            fig.add_vline(x=1, line_dash="dot", line_color=GRAY)
            fig.update_layout(title="Rising terms (share in last 3 days ÷ share before)", xaxis_title="Growth ratio",
                              height=480, **LAYOUT)
            c1.plotly_chart(fig, width="stretch", key="chart_2")
        else:
            c1.info("Not enough days of data to compute rising terms.")
        c2.dataframe(R["rising"], hide_index=True, width="stretch", height=480)
        st.plotly_chart(daily_chart(R), width="stretch", key="chart_3")

        st.markdown("#### Most relevant posts per keyword (TF-IDF + cosine)")
        kw = st.selectbox("Keyword", list(R["queries"]))
        top = R["scored"].nlargest(10, f"score::{kw}")[["clean", f"score::{kw}", "created_at", "likes"]]
        st.dataframe(top.rename(columns={"clean": "post", f"score::{kw}": "cosine"}), hide_index=True, width="stretch")

# ================================================================= THEMES
with tabs[3]:
    if R is None:
        need_results()
    else:
        th = R["themes"].copy()
        t = th.sort_values("positive")
        tot = t[["negative", "neutral", "positive"]].sum(axis=1)
        fig = go.Figure()
        for lab in ["negative", "neutral", "positive"]:
            fig.add_trace(go.Bar(y=t.theme, x=t[lab] / tot, name=lab, orientation="h", marker_color=SENT[lab],
                                 marker_line=dict(color="white", width=2),
                                 hovertemplate="%{y}<br>" + lab + ": %{x:.0%}<extra></extra>"))
        fig.update_layout(barmode="stack", title=f"Sentiment by theme · model: {R['sentiment_model']}",
                          xaxis=dict(tickformat=".0%"), height=60 + 45 * len(t), **LAYOUT)
        st.plotly_chart(fig, width="stretch", key="chart_4")
        show = th[["theme", "posts", "growth", "positive", "negative", "greenwash_share", "product", "stock_units", "decision"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "positive": st.column_config.ProgressColumn("positive", format="percent", min_value=0, max_value=1),
            "negative": st.column_config.NumberColumn("negative", format="percent"),
            "growth": st.column_config.NumberColumn("growth ×", format="%.2f"),
            "greenwash_share": st.column_config.NumberColumn("greenwash", format="percent")})
        st.download_button("Download theme table (CSV)", csv_bytes(th.drop(columns=["evidence", "evidence_positive"])), "theme_table.csv")
        for r in th.itertuples():
            with st.expander(f"{DECISION_ICON.get(r.decision, '')} {r.theme} — {r.posts} posts"):
                st.markdown(f"**Top words:** {r.top_words}")
                for e in r.evidence:
                    st.markdown(f"> {e}")
        st.caption(f"{R['topics']['method']} chose k = {R['topics']['k']} by NPMI coherence:")
        st.dataframe(R["topics"]["coherence_table"].T, width="stretch")

# ================================================================= REVIEWS
with tabs[4]:
    if R is None:
        need_results()
    elif "nps" not in R:
        st.info("No reviews were provided. Add a review export on the Setup tab.")
    else:
        nps = R["nps"].sort_values("est_nps_excl_fakes")
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(y=nps["product"], x=nps.est_nps_excl_fakes, orientation="h",
                               marker_color=[RED if v < 0 else BLUE for v in nps.est_nps_excl_fakes],
                               text=[f"{v:.1f}" for v in nps.est_nps_excl_fakes], textposition="outside",
                               hovertemplate="%{y}<br>est. NPS %{x}<extra></extra>"))
        lo, hi = float(nps.est_nps_excl_fakes.min()), float(nps.est_nps_excl_fakes.max())
        fig.update_layout(title="Estimated NPS (5★ promoters − 1–3★ detractors), fakes removed", height=420,
                          xaxis=dict(title="Est. NPS", range=[min(lo, 0) - 12, max(hi, 0) + 12]), **LAYOUT)
        c1.plotly_chart(fig, width="stretch", key="chart_5")
        a = R["aspect_neg"]
        if len(a):
            worst = a.stack().idxmax()
            fig = go.Figure(go.Heatmap(z=a.values, x=[c.replace("_", " ") for c in a.columns], y=a.index, zmin=0,
                                       zmax=max(0.5, float(np.nanmax(a.values))), colorscale=[[0, "#fcf1f0"], [1, "#b42f2f"]],
                                       text=a.values, texttemplate="%{text:.0%}",
                                       hovertemplate="%{y}<br>%{x}: %{z:.0%} negative<extra></extra>",
                                       colorbar=dict(tickformat=".0%")))
            fig.update_layout(title=f"Share of negative mentions — worst: {worst[1].replace('_', ' ')} of {worst[0]}",
                              height=420, **LAYOUT)
            c2.plotly_chart(fig, width="stretch", key="chart_6")
        st.caption("Real NPS needs the 'how likely are you to recommend us (0–10)' survey question; this is a proxy from stars.")
        st.dataframe(R["nps"], hide_index=True, width="stretch")
        rs = R["reviews_scored"]
        c1, c2 = st.columns(2)
        c1.markdown(f"#### 🚩 Suspected fake reviews ({int(rs.fake_flag.sum())})")
        c1.caption("≥5 near-identical 5★ reviews, mostly posted on the same day (MinHash).")
        c1.dataframe(rs[rs.fake_flag][["sku", "date", "stars", "text"]], hide_index=True, width="stretch")
        c2.markdown(f"#### 🌿 Greenwashing complaints ({int(rs.greenwash_flag.sum())})")
        c2.caption("An eco claim next to doubt words (but, claims, not really, never…).")
        c2.dataframe(rs[rs.greenwash_flag][["sku", "date", "stars", "text"]], hide_index=True, width="stretch")

# ================================================================= SALES LINK
with tabs[5]:
    if R is None:
        need_results()
    elif "sales_link" not in R:
        st.info("Needs both reviews and weekly sales.")
    else:
        L = R["sales_link"].sort_values("pearson_r")
        fig = go.Figure(go.Bar(y=L["product"], x=L.pearson_r, orientation="h",
                               marker_color=[RED if v < 0 else BLUE for v in L.pearson_r.fillna(0)],
                               text=[f"{v:.2f}" for v in L.pearson_r], textposition="outside"))
        fig.update_layout(title="Correlation: weekly positive reviews vs weekly units sold", xaxis=dict(range=[-1, 1.15]),
                          xaxis_title="Pearson r", height=420, **LAYOUT)
        st.plotly_chart(fig, width="stretch", key="chart_7")
        st.dataframe(R["sales_link"], hide_index=True, width="stretch")
        st.warning("This is an **estimate of association**, not a forecast: a dozen weeks of data cannot prove that reviews "
                   "cause sales. On the sample data the link was built into some products to demonstrate the method.")

# ================================================================= CAMPAIGN STUDIO
with tabs[6]:
    if R is None:
        need_results()
    else:
        camps = R["campaigns"]
        if not camps:
            st.info("No theme passed the filters for promotion. Check the alerts on the Overview tab.")
        for i, c in enumerate(camps):
            with st.container(border=True):
                st.markdown(f"### 📣 {c['hook']}")
                st.caption(f"{c['product']} · {c['stock_units']} in stock · {c['channels']} · copy by {c['source']}")
                st.markdown(f"**Evidence:** {c['mentions']} relevant posts · {c['positive_share']:.0%} positive · "
                            f"peak growth {c['growth']}×")
                for e in c["evidence"]:
                    st.markdown(f"> {e}")
                c1, c2 = st.columns(2)
                c["linkedin"] = c1.text_area("LinkedIn (founder)", c["linkedin"], height=170, key=f"li_{i}")
                c["instagram"] = c2.text_area("Instagram caption", c["instagram"], height=170, key=f"ig_{i}")
                st.markdown("**Blog ideas:** " + " · ".join(c["blog_titles"]))
        th = R["themes"]
        for r in th[th.decision.isin(["Fix-first alert", "Clarify eco claims"])].itertuples():
            tip = ("Post a service update, reply to complaints within 24h, and pause ads on the affected channel."
                   if r.decision == "Fix-first alert" else "Publish certificates / material details before promoting eco claims.")
            st.error(f"**{r.decision}: {r.theme}** — {r.posts} posts, {r.negative:.0%} negative. {tip}")
        st.markdown("#### 🗓️ Two-week content calendar")
        cal = st.data_editor(R["calendar"], num_rows="dynamic", width="stretch", key="cal")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("campaigns.json", json.dumps(camps, indent=2, ensure_ascii=False, default=str))
            z.writestr("content_calendar.csv", cal.to_csv(index=False))
            z.writestr("theme_table.csv", th.drop(columns=["evidence", "evidence_positive"]).to_csv(index=False))
            z.writestr("rising_terms.csv", R["rising"].to_csv(index=False))
            if "nps" in R:
                z.writestr("estimated_nps.csv", R["nps"].to_csv(index=False))
                z.writestr("reviews_scored.csv", R["reviews_scored"].to_csv(index=False))
            z.writestr("evaluation.csv", R["evaluation"].to_csv(index=False))
        st.download_button("⬇️ Download everything (ZIP)", buf.getvalue(), "marketpulse_outputs.zip", "application/zip",
                           type="primary")

# ================================================================= EVALUATION
with tabs[7]:
    if R is None:
        need_results()
    else:
        st.dataframe(R["evaluation"], hide_index=True, width="stretch")
        if R.get("sentiment_eval") is not None:
            st.markdown("#### Sentiment models on held-out labelled posts")
            st.dataframe(R["sentiment_eval"], hide_index=True, width="stretch")
        if "gt_theme" in R["relevant"]:
            st.info("Ground-truth scores (duplicates, Precision@10, purity, fake reviews) are available because the "
                    "**sample data is synthetic**. With real data, hand-label a few hundred posts to compute them.")
        st.markdown("""
**How to read this**
- *Duplicates:* recall is what matters; 'false positives' on synthetic data are genuine posts with identical wording.
- *Sentiment:* VADER needs no labels but misses context and sarcasm; a model trained on review stars transfers badly
  to social posts (**domain shift**); a model trained on labelled posts scores very high here only because the synthetic
  posts reuse sentence templates — expect lower on real posts.
- *Topics:* NMF on TF-IDF gave cleaner themes on short posts than LDA (purity ≈ 0.84 vs ≈ 0.65 on the sample),
  and runs in under a second.
""")

# ================================================================= ABOUT
with tabs[8]:
    st.markdown("""
### How it works
| Step | Technique | Syllabus |
|---|---|---|
| Collect | Official X API or uploaded CSV | Units 1–2 |
| Clean + dedupe | Regex cleaning; MinHash + LSH near-duplicate removal (Jaccard ≥ 0.8) | Unit 6 |
| Preprocess | Tokenise, stop words (negations kept), Porter stemming | Units 2, 5 |
| Relevance | TF-IDF + cosine similarity with query expansion | Unit 5 |
| Trends | Share of posts in last 3 days ÷ share before | Content mining |
| Themes | NMF (default) or LDA, k chosen by NPMI coherence | Unit 6 |
| Sentiment | VADER lexicon vs TF-IDF + logistic regression | Unit 6 |
| Reviews | Fake-review clusters, est. NPS, aspect sentiment, greenwashing rule | Unit 6 |
| Decide | Stock filter and decision rules (business rules, not mining) | — |
| Create | Template copy, optional LLM grounded in the mined facts | Output layer |

### Data and ethics
- No scraping: posts come from the **official X API**, your own uploads, or **synthetic sample data**.
- Data minimisation: only text, date and like counts are kept — no usernames, profiles or locations.
- Generated copy never quotes users by name or invents testimonials; every campaign shows its evidence.
- Sample data is fictional (brand "EcoNest") and generated with a fixed seed.

*Built by Shaunak Wanelkar for the Web Data Mining laboratory (Data Science minor, VJTI).*
""")
