"""MarketPulse mining pipeline.

Every function here is a plain Python function so the Streamlit app (app.py), a notebook or a
script can call it. The steps follow the project roadmap:

  1-2  clean + near-duplicate removal (MinHash/LSH)       Unit 6
  3    preprocessing (stop words, Porter stemming)          Unit 5
  4    relevance ranking (TF-IDF + cosine, query expansion)   Unit 5
  5    rising terms
  6    themes (LDA, NPMI coherence)                         Unit 6
  7    sentiment (VADER / logistic regression)               Unit 6
  8    reviews: fake-review alert, estimated NPS, aspects, greenwashing
  9    theme table      10 stock filter      11 sales link
  12   campaign generation (template, optional LLM)     14 evaluation
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.stem import PorterStemmer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
LABELS = ["positive", "neutral", "negative"]

# --------------------------------------------------------------------------------------------
# Input schemas (used by the app for validation and template downloads)
# --------------------------------------------------------------------------------------------
SCHEMAS = {
    "posts": {"required": ["text", "created_at"], "optional": ["post_id", "likes"]},
    "inventory": {"required": ["sku", "product", "stock_units"], "optional": ["category", "channels", "tags"]},
    "reviews": {"required": ["sku", "date", "stars", "text"], "optional": ["review_id"]},
    "sales": {"required": ["sku", "week_start", "units"], "optional": []},
    "labelled": {"required": ["text", "label"], "optional": []},
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate(df: pd.DataFrame, kind: str) -> list[str]:
    missing = [c for c in SCHEMAS[kind]["required"] if c not in df.columns]
    return [f"{kind}: missing column(s) {', '.join(missing)}"] if missing else []


def template(kind: str) -> pd.DataFrame:
    examples = {
        "posts": [dict(post_id="1", created_at="2026-10-20", text="Loving reusable fabric gift wrap this diwali", likes=4)],
        "inventory": [dict(sku="P01", product="Reusable Fabric Gift Wrap", category="Gifting",
                           channels="Amazon; Sustainable marketplace", stock_units=400, tags="gift wrap wrapping")],
        "reviews": [dict(review_id="r1", sku="P01", date="2026-10-12", stars=5, text="Lovely plastic free packaging.")],
        "sales": [dict(sku="P01", week_start="2026-10-12", units=40)],
        "labelled": [dict(text="Package arrived damaged", label="negative")],
    }
    return pd.DataFrame(examples[kind])


def parse_keywords(text: str) -> dict[str, str]:
    """Lines like 'eco diwali: diya candle gift' -> {'eco diwali': 'eco diwali diya candle gift'}."""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        kw, _, extra = line.partition(":")
        kw = kw.strip()
        out[kw] = f"{kw} {extra.strip()}".strip()
    return out


# --------------------------------------------------------------------------------------------
# Optional: official X API (recent search). Never scrape X - its terms prohibit it.
# --------------------------------------------------------------------------------------------
def fetch_x_posts(bearer_token: str, keywords: list[str], per_keyword: int = 100) -> pd.DataFrame:
    import requests

    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    rows = []
    for kw in keywords:
        params = {"query": f'"{kw}" lang:en -is:retweet', "max_results": 100,
                  "tweet.fields": "created_at,public_metrics"}
        fetched, token = 0, None
        while fetched < per_keyword:
            if token:
                params["next_token"] = token
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"X API error {r.status_code}: {r.text[:200]}")
            js = r.json()
            for t in js.get("data", []):  # store only what we need (no usernames)
                rows.append(dict(post_id=t["id"], created_at=t["created_at"][:10], text=t["text"],
                                 likes=t.get("public_metrics", {}).get("like_count", 0)))
            fetched += len(js.get("data", []))
            token = js.get("meta", {}).get("next_token")
            if not token:
                break
    return pd.DataFrame(rows).drop_duplicates("post_id") if rows else pd.DataFrame(columns=["post_id", "created_at", "text", "likes"])


# --------------------------------------------------------------------------------------------
# Steps 1-2: cleaning and near-duplicate removal
# --------------------------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^RT @\w+:\s*")
EMOJI_RE = re.compile(r"[^\w\s.,!?'#-]")


def clean_post(text) -> str:
    t = RT_RE.sub("", str(text))
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = re.sub(r"#(\w+)", lambda m: re.sub(r"([a-z])([A-Z])", r"\1 \2", m.group(1)), t)
    t = EMOJI_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def text_for_sentiment(text) -> str:
    t = RT_RE.sub("", str(text))
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(text: str, k: int = 3) -> set[str]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=SEED)
    for s in shingles(text):
        m.update(s.encode("utf8"))
    return m


def dedupe(df: pd.DataFrame, text_col: str, id_col: str, threshold: float = 0.8):
    """Keep the earliest item of each near-duplicate group (estimated Jaccard >= threshold)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept, removed = [], {}
    for row in df.sort_values(["created_at", id_col]).itertuples():
        mh = minhash(getattr(row, text_col))
        hits = lsh.query(mh)
        rid = getattr(row, id_col)
        if hits:
            removed[rid] = hits[0]
        else:
            lsh.insert(rid, mh)
            kept.append(rid)
    return df[df[id_col].isin(kept)].copy(), removed


# --------------------------------------------------------------------------------------------
# Step 3: preprocessing
# --------------------------------------------------------------------------------------------
NEGATIONS = {"not", "no", "nor", "never", "nothing", "none", "cannot"}
FILLER_WORDS = {
    "love", "got", "keeps", "suggestions", "need", "think", "happy", "today", "update", "honestly", "okay", "tell",
    "list", "pics", "later", "thread", "rant", "worth", "sharing", "lesson", "learnt", "mood", "right", "alone",
    "noted", "anyone", "rate", "ten", "quick", "real", "talk", "story", "time", "morning", "thought", "weekend",
    "hot", "take", "note", "self", "family", "chat", "says", "friendly", "reminder", "check", "diaries",
    "apparently", "record", "saying", "win", "small", "gonna", "lie", "folks", "day", "prep", "post", "adding",
    "end", "updates", "soon", "help", "fyi", "going", "week", "buy", "tips", "lovely", "case", "helps"}
STOP = (set(ENGLISH_STOP_WORDS) - NEGATIONS) | {"amp", "rt", "via", "just", "really", "so", "im"} | FILLER_WORDS


class Preprocessor:
    """Tokenise, drop stop words (keeping negations), Porter-stem; remember a readable word per stem."""

    def __init__(self, extra_stop: set[str] | None = None):
        self.stemmer = PorterStemmer()
        self.stop = STOP | (extra_stop or set())
        self.display = defaultdict(Counter)

    def __call__(self, text: str) -> list[str]:
        out = []
        for t in re.findall(r"[a-z]+", str(text).lower()):
            if len(t) < 2 or t in self.stop:
                continue
            s = self.stemmer.stem(t)
            self.display[s][t] += 1
            out.append(s)
        return out

    def show(self, stem: str) -> str:
        return self.display[stem].most_common(1)[0][0] if stem in self.display else stem


# --------------------------------------------------------------------------------------------
# Step 4: relevance ranking
# --------------------------------------------------------------------------------------------
def rank_relevance(df: pd.DataFrame, queries: dict[str, str], prep: Preprocessor, threshold: float):
    q_docs = {k: " ".join(prep(v)) for k, v in queries.items()}
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2 if len(df) > 50 else 1)
    X = vec.fit_transform(df["doc"])
    sims = cosine_similarity(X, vec.transform(list(q_docs.values())))
    df = df.copy()
    for i, k in enumerate(q_docs):
        df[f"score::{k}"] = sims[:, i]
    df["relevance"] = sims.max(axis=1)
    df["best_keyword"] = [list(q_docs)[i] for i in sims.argmax(axis=1)]
    return df, df[df.relevance >= threshold].copy()


# --------------------------------------------------------------------------------------------
# Step 5: rising terms
# --------------------------------------------------------------------------------------------
def rising_terms(rel: pd.DataFrame, prep: Preprocessor, recent_days: int = 3, min_total: int = 10) -> pd.DataFrame:
    day = pd.to_datetime(rel["created_at"])
    recent = day >= day.max() - pd.Timedelta(days=recent_days - 1)
    if recent.all() or (~recent).all():
        return pd.DataFrame(columns=["term", "mentions_recent", "mentions_before", "growth_ratio"])

    def share(mask):
        sub = rel[mask]
        c = Counter(t for toks in sub["tokens"] for t in set(toks))
        return {t: n / len(sub) for t, n in c.items()}, c

    s_new, c_new = share(recent)
    s_old, c_old = share(~recent)
    rows = []
    for t in set(c_new) | set(c_old):
        total = c_new.get(t, 0) + c_old.get(t, 0)
        if total < min_total:
            continue
        rows.append(dict(term=prep.show(t), mentions_recent=c_new.get(t, 0), mentions_before=c_old.get(t, 0),
                         growth_ratio=round((s_new.get(t, 0) + 0.005) / (s_old.get(t, 0) + 0.005), 2)))
    return pd.DataFrame(rows).sort_values("growth_ratio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Step 6: topic modeling (LDA + NPMI coherence)
# --------------------------------------------------------------------------------------------
def fit_topics(rel: pd.DataFrame, prep: Preprocessor, k_min: int = 6, k_max: int = 10, restarts: int = 3,
               method: str = "nmf"):
    """Topic model chosen by NPMI coherence over k_min..k_max.

    method="nmf": Non-negative Matrix Factorisation on TF-IDF (fast, deterministic, works well on short posts).
    method="lda": Latent Dirichlet Allocation on counts with a small alpha (one topic per post) and restarts.
    """
    min_df = 5 if len(rel) > 300 else 2
    if method == "nmf":
        vec = TfidfVectorizer(min_df=min_df, max_df=0.35, sublinear_tf=True)
    else:
        vec = CountVectorizer(min_df=min_df, max_df=0.35)
    X = vec.fit_transform(rel["doc"])
    vocab = np.array(vec.get_feature_names_out())
    Xb = (X > 0).astype(int).tocsc()
    n_docs = Xb.shape[0]
    dfreq = np.asarray(Xb.sum(axis=0)).ravel()

    def npmi(top_idx):
        scores = []
        for a, b in itertools.combinations(top_idx, 2):
            co = Xb[:, a].multiply(Xb[:, b]).sum()
            if co == 0:
                scores.append(-1.0)
                continue
            p_ab = co / n_docs
            scores.append(math.log(p_ab / ((dfreq[a] / n_docs) * (dfreq[b] / n_docs))) / -math.log(p_ab))
        return float(np.mean(scores))

    def fit(k, seed):
        if method == "nmf":
            return NMF(n_components=k, init="nndsvd", random_state=seed, max_iter=400).fit(X)
        return LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch", max_iter=50,
                                         doc_topic_prior=0.05, topic_word_prior=0.01).fit(X)

    k_max = max(k_min, min(k_max, len(vocab) // 3))
    coherence, best = [], None
    for k in range(k_min, k_max + 1):
        best_k = None
        for r in range(1 if method == "nmf" else restarts):
            m = fit(k, r)
            coh = np.mean([npmi(c.argsort()[::-1][:10]) for c in m.components_])
            if best_k is None or coh > best_k[0]:
                best_k = (coh, m)
        coherence.append(dict(k=k, npmi=round(best_k[0], 3)))
        if best is None or best_k[0] > best[0]:
            best = (best_k[0], best_k[1], k)

    coh, model, k = best
    comp = model.components_ + 1e-9
    topic_word = comp / comp.sum(axis=1, keepdims=True)
    distinct = topic_word * np.log(topic_word / topic_word.mean(axis=0, keepdims=True))
    labels, top_words = {}, {}
    for t in range(k):
        first = vocab[topic_word[t].argmax()]
        extra = [w for w in vocab[distinct[t].argsort()[::-1]] if w != first][:2]
        labels[t] = " / ".join(prep.show(w) for w in [first] + extra)
        top_words[t] = [prep.show(w) for w in vocab[topic_word[t].argsort()[::-1][:8]]]
    assign = model.transform(X).argmax(axis=1)
    return dict(method=method.upper(), k=k, coherence=coh, coherence_table=pd.DataFrame(coherence), vocab=vocab,
                topic_word=topic_word, labels=labels, top_words=top_words, assign=assign)


# --------------------------------------------------------------------------------------------
# Step 7: sentiment
# --------------------------------------------------------------------------------------------
_vader = SentimentIntensityAnalyzer()


def vader_label(text: str, th: float = 0.05) -> str:
    c = _vader.polarity_scores(text)["compound"]
    return "positive" if c >= th else "negative" if c <= -th else "neutral"


def train_logreg(texts, labels):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(vec.fit_transform(texts), labels)
    return lambda xs: clf.predict(vec.transform(list(xs)))


def star_label(s) -> str:
    return "positive" if s >= 4 else "neutral" if s == 3 else "negative"


def sentiment_models(labelled: pd.DataFrame | None, reviews: pd.DataFrame | None):
    """Builds the available models and evaluates them on a held-out part of the labelled posts."""
    models = {"VADER (lexicon, no training)": lambda xs: [vader_label(text_for_sentiment(x)) for x in xs]}
    if reviews is not None and len(reviews) >= 50:
        models["LogReg trained on review stars"] = train_logreg(reviews["text"], reviews["stars"].map(star_label))
    evaluation, test = None, None
    if labelled is not None and len(labelled) >= 60:
        lab = labelled.assign(label=labelled["label"].str.lower().str.strip())
        lab = lab[lab.label.isin(LABELS)]
        train, test = train_test_split(lab, test_size=0.3, random_state=SEED, stratify=lab.label)
        models["LogReg trained on labelled posts"] = train_logreg(train["text"].map(clean_post), train["label"])
        rows = []
        for name, f in models.items():
            pred = f(test["text"] if name.startswith("VADER") else test["text"].map(clean_post))
            rows.append(dict(model=name, accuracy=round(accuracy_score(test.label, pred), 3),
                             macro_f1=round(f1_score(test.label, pred, average="macro"), 3)))
        evaluation = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return models, evaluation, test


# --------------------------------------------------------------------------------------------
# Step 8: reviews
# --------------------------------------------------------------------------------------------
ECO = r"(plastic.?free|compostable|biodegradable|eco.?friendly|sustainable)"
DOUBT = r"(\bbut\b|not really|\bclaims?\b|\bnever\b|\bfake\b|greenwash|so called)"
ASPECT_LEX = {
    "quality": r"\b(quality|well made|finished|sturdy|cheap|broke)\b",
    "packaging": r"\b(packaging|packed)\b",
    "delivery": r"\b(delivery|delivered|arrived|late)\b",
    "price": r"\b(price|value|expensive|overpriced|pricey)\b",
    "durability": r"\b(lasts|durability|wore out|stopped working|bristles|leaks?|going strong)\b",
    "eco_claim": r"\b(plastic free|compostable|biodegradable|eco friendly)\b",
}
DOMAIN_LEX = {"late": -1.8, "delayed": -1.8, "torn": -1.8, "leaks": -2.0, "leaking": -2.0, "overpriced": -2.2,
              "expensive": -1.3, "pricey": -0.8, "broke": -1.8, "wore": -1.2, "stopped": -1.2, "quick": 1.2,
              "sturdy": 1.5, "lasts": 1.2}
_vader_dom = SentimentIntensityAnalyzer()
_vader_dom.lexicon.update(DOMAIN_LEX)


def greenwash(text: str) -> bool:
    t = str(text).lower()
    return bool(re.search(ECO, t) and re.search(DOUBT, t))


def analyse_reviews(reviews: pd.DataFrame, inventory: pd.DataFrame):
    rev = reviews.copy()
    if "review_id" not in rev:
        rev["review_id"] = [f"r{i}" for i in range(len(rev))]
    rev["review_id"] = rev["review_id"].astype(str)
    rev["norm"] = rev["text"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)

    # Fake-review alert: cluster of >=5 near-identical 5-star reviews, mostly on one day
    lsh, mhs = MinHashLSH(threshold=0.8, num_perm=128), {}
    for r in rev.itertuples():
        mhs[r.review_id] = minhash(r.norm)
        lsh.insert(r.review_id, mhs[r.review_id])
    rev["fake_flag"] = False
    seen = set()
    for r in rev.itertuples():
        if r.review_id in seen:
            continue
        cluster = rev[rev.review_id.isin(lsh.query(mhs[r.review_id])) & (rev.sku == r.sku)]
        seen |= set(cluster.review_id)
        if len(cluster) >= 5 and (cluster.stars == 5).all() and cluster["date"].value_counts(normalize=True).iloc[0] >= 0.8:
            rev.loc[cluster.index, "fake_flag"] = True

    rev["greenwash_flag"] = rev["text"].map(greenwash)

    def est_nps(d):
        return round(100 * ((d.stars == 5).mean() - (d.stars <= 3).mean()), 1) if len(d) else np.nan

    names = dict(zip(inventory.sku.astype(str), inventory["product"]))
    nps = pd.DataFrame([dict(sku=s, product=names.get(s, s), reviews=len(d), avg_stars=round(d.stars.mean(), 2),
                             est_nps_all=est_nps(d), est_nps_excl_fakes=est_nps(d[~d.fake_flag]),
                             fake_reviews=int(d.fake_flag.sum()), greenwash_complaints=int(d.greenwash_flag.sum()))
                        for s, d in rev.groupby("sku")])

    rows = []
    for r in rev[~rev.fake_flag].itertuples():
        for sent in re.split(r"[.!?]+", str(r.text)):
            s = sent.strip().lower()
            if not s:
                continue
            for asp, pat in ASPECT_LEX.items():
                if re.search(pat, s):
                    score = -0.6 if (asp == "eco_claim" and greenwash(s)) else _vader_dom.polarity_scores(s)["compound"]
                    rows.append(dict(sku=r.sku, aspect=asp, compound=score))
    aspects = pd.DataFrame(rows, columns=["sku", "aspect", "compound"])
    neg_share = (aspects.assign(neg=aspects.compound <= -0.05).groupby(["sku", "aspect"])["neg"].mean().unstack())
    neg_share.index = neg_share.index.map(lambda s: names.get(s, s))
    return rev.drop(columns=["norm"]), nps, neg_share


# --------------------------------------------------------------------------------------------
# Steps 9-10: theme table and decisions
# --------------------------------------------------------------------------------------------
def product_tags(inventory: pd.DataFrame, stemmer: PorterStemmer) -> dict[str, list[str]]:
    tags = {}
    for r in inventory.itertuples():
        words = str(getattr(r, "tags", "") or "")
        if not words.strip() or words == "nan":
            words = re.sub(r"\(.*?\)|\d+\w*", " ", str(r.product))
        tags[str(r.sku)] = [stemmer.stem(w) for w in re.findall(r"[a-z]+", words.lower()) if w not in STOP]
    return tags


def build_theme_table(rel, topics, inventory, prep, min_stock=100, link_threshold=0.04):
    vocab_index = {w: i for i, w in enumerate(topics["vocab"])}
    tags = product_tags(inventory, prep.stemmer)
    day = pd.to_datetime(rel["created_at"])
    recent_start = day.max() - pd.Timedelta(days=2)          # last 3 calendar days
    n_recent_days = 3
    n_old_days = max(1, (recent_start - day.min()).days)
    rows = []
    for t, g in rel.groupby("topic"):
        tw = topics["topic_word"][t]
        scores = {s: sum(tw[vocab_index[x]] for x in tg if x in vocab_index) for s, tg in tags.items()}
        sku, sc = max(scores.items(), key=lambda kv: kv[1]) if scores else (None, 0)
        sku = sku if sc >= link_threshold else None
        gd = pd.to_datetime(g["created_at"])
        recent, earlier = (gd >= recent_start).sum() / n_recent_days, (gd < recent_start).sum() / n_old_days
        shares = g["sentiment"].value_counts(normalize=True)
        rows.append(dict(topic=t, theme=topics["labels"][t], top_words=", ".join(topics["top_words"][t]), posts=len(g),
                         growth=round(recent / max(earlier, 0.1), 2), positive=shares.get("positive", 0.0),
                         neutral=shares.get("neutral", 0.0), negative=shares.get("negative", 0.0),
                         greenwash_share=round(g["greenwash_flag"].mean(), 2), avg_likes=round(g["likes"].mean(), 1),
                         sku=sku, evidence=g.sort_values("likes", ascending=False)["clean"].head(3).tolist(),
                         evidence_positive=g[g.sentiment == "positive"].sort_values("likes", ascending=False)["clean"].head(3).tolist()))
    inv = inventory.assign(sku=inventory.sku.astype(str))
    cols = [c for c in ["sku", "product", "category", "channels", "stock_units"] if c in inv]
    themes = pd.DataFrame(rows).merge(inv[cols], on="sku", how="left")

    def decide(r):
        if r.greenwash_share >= 0.25:
            return "Clarify eco claims"
        if r.negative >= 0.5:
            return "Fix-first alert"
        if pd.isna(r.sku):
            return "Brand content"
        if r.stock_units < min_stock:
            return "Restock before promoting"
        if r.positive >= 0.45:
            return "Promote"
        return "Monitor"

    themes["decision"] = themes.apply(decide, axis=1)
    return themes.sort_values("posts", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Step 11: sales link (estimate)
# --------------------------------------------------------------------------------------------
def sales_link(reviews_scored: pd.DataFrame, sales: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["week_start"] = pd.to_datetime(s["week_start"])
    r = reviews_scored[~reviews_scored.fake_flag & (reviews_scored.stars >= 4)].copy()
    r["date"] = pd.to_datetime(r["date"])
    out = []
    names = dict(zip(inventory.sku.astype(str), inventory["product"]))
    for sku, g in s.groupby("sku"):
        rs = r[r.sku == sku]
        pos = [((rs.date >= w) & (rs.date < w + pd.Timedelta(days=7))).sum() for w in g.week_start]
        rr = np.corrcoef(pos, g.units)[0, 1] if len(g) >= 4 and np.std(pos) > 0 and g.units.std() > 0 else np.nan
        out.append(dict(sku=sku, product=names.get(str(sku), sku), weeks=len(g), avg_weekly_units=round(g.units.mean(), 1),
                        pearson_r=round(rr, 2) if not np.isnan(rr) else np.nan))
    link = pd.DataFrame(out)
    link["reading"] = pd.cut(link.pearson_r, [-1.01, -0.3, 0.3, 0.6, 1.01],
                             labels=["negative", "weak / none", "moderate positive", "strong positive"])
    return link


# --------------------------------------------------------------------------------------------
# Step 12: campaigns
# --------------------------------------------------------------------------------------------
HOOKS = {"Gifting": ["Give gifts, not garbage this {occ}", "Wrap {occ}, not the planet", "{occ} gifting that grows back"],
         "Home": ["Light up {occ}, not the air", "A cleaner glow for {occ}"],
         "Personal Care": ["Small swap, big difference"], "Kitchen": ["Refill, reuse, repeat"],
         "Fashion": ["Carry less plastic"]}


CATEGORY_HINTS = {"Gifting": ["gift", "wrap", "card", "hamper", "greeting"], "Home": ["candle", "diya", "lamp", "decor"],
                  "Personal Care": ["brush", "toothbrush", "soap", "shampoo", "comb"],
                  "Kitchen": ["bottle", "scrubber", "cutlery", "straw", "container"], "Fashion": ["bag", "tote", "shirt"]}
GENERIC_HOOKS = ["Make the switch this {occ}", "Celebrate {occ} the low-waste way", "Small swaps for a greener {occ}"]


def infer_category(row) -> str:
    cat = row.get("category")
    if isinstance(cat, str) and cat.strip():
        return cat.strip()
    name = str(row.get("product", "")).lower()
    for c, words in CATEGORY_HINTS.items():
        if any(re.search(rf"\b{w}", name) for w in words):
            return c
    return ""


def template_campaigns(themes: pd.DataFrame, brand: str, occasion: str) -> list[dict]:
    out = []
    for sku, rows in themes[themes.decision == "Promote"].groupby("sku"):
        r0 = rows.iloc[0]
        words = []
        for tw in rows.top_words:
            words += tw.split(", ")
        kw = ", ".join(list(dict.fromkeys(w for w in words if w not in {"eco", "sustainable", "no", "plastic", "free"}))[:3])
        n = int(rows.posts.sum())
        pos = float((rows.positive * rows.posts).sum() / n)
        growth = float(rows.growth.max())
        prod_l = str(r0["product"]).split("(")[0].strip().lower()
        category = infer_category(r0)
        hook = random.Random(str(sku)).choice(HOOKS.get(category, GENERIC_HOOKS)).format(occ=occasion)
        channels = str(r0.get("channels", "") or "").replace(";", ",")
        out.append(dict(
            sku=sku, product=r0["product"], stock_units=int(r0["stock_units"]), channels=channels, hook=hook,
            blog_titles=[f"{n} people talked about {kw.split(',')[0]} this fortnight: why {occasion} is going green",
                         f"How to choose a {prod_l} that is actually sustainable",
                         f"From the founder of {brand}: what customers taught us about {kw.split(',')[0]}"],
            linkedin=(f"{n} posts in the last two weeks were about {kw}, and {pos:.0%} of them were positive. "
                      f"Conversations grew {growth:.1f}x in the last 3 days. That's why this {occasion} {brand} is "
                      f"putting our {prod_l} front and centre. Made to be used again, not thrown away."
                      + (f" Available on {channels}." if channels else "")),
            instagram=f"{hook} ✨ Meet our {prod_l}, made for your {occasion} plans. #{occasion.replace(' ', '')} #EcoDiwali #SustainableLiving #PlasticFree",
            evidence=[e for ev in rows.evidence_positive for e in ev][:3], mentions=n, positive_share=round(pos, 2), growth=growth,
            source="template"))
    return out


LLM_PROMPT = """You are the marketing assistant for {brand}, a small sustainable D2C brand.
Use ONLY the facts below. Do not invent statistics, testimonials or certifications.
Facts (JSON): {facts}
Return JSON with keys: hook (max 8 words), blog_titles (list of 3), linkedin (founder voice, max 90 words),
instagram (caption with 4 hashtags)."""


def llm_campaigns(campaigns: list[dict], api_key: str, brand: str, model: str = "claude-sonnet-5-5") -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    out = []
    for c in campaigns:
        facts = {k: c[k] for k in ["product", "stock_units", "channels", "mentions", "positive_share", "growth", "evidence"]}
        msg = client.messages.create(model=model, max_tokens=700, messages=[
            {"role": "user", "content": LLM_PROMPT.format(brand=brand, facts=json.dumps(facts))}])
        text = msg.content[0].text
        try:
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            out.append({**c, **{k: data[k] for k in ["hook", "blog_titles", "linkedin", "instagram"] if k in data},
                        "source": "LLM"})
        except (ValueError, KeyError):
            out.append(c)
    return out


def content_calendar(campaigns: list[dict], has_delivery_alert: bool, start: pd.Timestamp) -> pd.DataFrame:
    slots = ["Instagram", "LinkedIn", "Blog", "Instagram", "LinkedIn", "Instagram", "Rest"]
    rows = []
    for i in range(14):
        day, platform = (start + pd.Timedelta(days=i)).date(), slots[i % 7]
        if platform == "Rest" or not campaigns:
            rows.append(dict(date=day, platform="—", content="No post (review metrics)"))
            continue
        if i == 1 and has_delivery_alert:
            rows.append(dict(date=day, platform="Instagram + X", content="Service update: what we're doing about delivery delays"))
            continue
        c = campaigns[i % len(campaigns)]
        content = {"Instagram": c["instagram"], "LinkedIn": "Founder post: " + c["hook"], "Blog": c["blog_titles"][i % 3]}[platform]
        rows.append(dict(date=day, platform=platform, content=content))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
def run_pipeline(posts, inventory, keywords, reviews=None, sales=None, labelled=None, settings=None, progress=None):
    """Runs every step. `progress(msg)` is an optional callback for the UI."""
    st_ = dict(dedupe_threshold=0.8, relevance_threshold=0.10, k_min=6, k_max=10, restarts=3, min_stock=100,
               topic_method="nmf",
               sentiment="auto", brand="EcoNest", occasion="Diwali")
    st_.update(settings or {})
    say = progress or (lambda m: None)
    R = {"settings": st_}

    posts = posts.copy()
    if "post_id" not in posts:
        posts["post_id"] = [f"p{i}" for i in range(len(posts))]
    posts["post_id"] = posts["post_id"].astype(str)
    if "likes" not in posts:
        posts["likes"] = 0
    posts["likes"] = pd.to_numeric(posts["likes"], errors="coerce").fillna(0)
    posts["created_at"] = pd.to_datetime(posts["created_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    posts = posts.dropna(subset=["created_at", "text"])
    inventory = inventory.copy()
    inventory["sku"] = inventory["sku"].astype(str)
    inventory["stock_units"] = pd.to_numeric(inventory["stock_units"], errors="coerce").fillna(0).astype(int)

    say("Cleaning posts and removing near-duplicates (MinHash)…")
    posts["clean"] = posts["text"].map(clean_post)
    posts["sent_text"] = posts["text"].map(text_for_sentiment)
    unique, removed = dedupe(posts, "clean", "post_id", st_["dedupe_threshold"])
    R.update(raw_posts=len(posts), unique_posts=len(unique), removed_map=removed)

    say("Preprocessing text…")
    prep = Preprocessor()
    unique["tokens"] = unique["clean"].map(prep)
    unique["doc"] = unique["tokens"].map(" ".join)

    say("Ranking posts by relevance (TF-IDF + cosine)…")
    queries = parse_keywords(keywords)
    scored, rel = rank_relevance(unique, queries, prep, st_["relevance_threshold"])
    if len(rel) < 30:
        raise ValueError(f"Only {len(rel)} posts passed the relevance filter. Add more posts, broaden the keywords "
                         f"or lower the relevance threshold.")
    R.update(queries=queries, scored=scored)

    say("Finding rising terms…")
    R["rising"] = rising_terms(rel, prep, min_total=10 if len(rel) > 300 else 3)

    say(f"Finding themes with {st_['topic_method'].upper()}…")
    topics = fit_topics(rel, prep, st_["k_min"], st_["k_max"], st_["restarts"], st_["topic_method"])
    rel["topic"] = topics["assign"]
    rel["topic_label"] = rel["topic"].map(topics["labels"])
    R["topics"] = topics

    say("Classifying sentiment…")
    rv = None
    if reviews is not None:
        reviews = reviews.copy()
        reviews["sku"] = reviews["sku"].astype(str)
        reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")
        reviews = reviews.dropna(subset=["stars", "text"])
        rv = reviews
    models, sent_eval, _ = sentiment_models(labelled, rv)
    choice = st_["sentiment"]
    if choice == "auto" or choice not in models:
        choice = sent_eval.iloc[0]["model"] if sent_eval is not None else "VADER (lexicon, no training)"
    f = models[choice]
    rel["sentiment"] = f(rel["sent_text"] if choice.startswith("VADER") else rel["clean"])
    rel["greenwash_flag"] = rel["clean"].map(greenwash)
    R.update(sentiment_model=choice, sentiment_eval=sent_eval)

    if rv is not None and len(rv):
        say("Analysing reviews (fake reviews, est. NPS, aspects, greenwashing)…")
        R["reviews_scored"], R["nps"], R["aspect_neg"] = analyse_reviews(rv, inventory)
        if sales is not None and len(sales):
            say("Estimating the sales link…")
            sales = sales.copy()
            sales["sku"] = sales["sku"].astype(str)
            sales["units"] = pd.to_numeric(sales["units"], errors="coerce").fillna(0)
            R["sales_link"] = sales_link(R["reviews_scored"], sales, inventory)

    say("Building the theme table and applying the stock filter…")
    themes = build_theme_table(rel, topics, inventory, prep, st_["min_stock"])
    R.update(relevant=rel, themes=themes)

    say("Writing campaign ideas…")
    camps = template_campaigns(themes, st_["brand"], st_["occasion"])
    if st_.get("llm_key"):
        try:
            camps = llm_campaigns(camps, st_["llm_key"], st_["brand"])
        except Exception as e:  # keep the template copy if the API fails
            R["llm_error"] = str(e)
    R["campaigns"] = camps
    start = pd.to_datetime(rel["created_at"]).max() + pd.Timedelta(days=1)
    R["calendar"] = content_calendar(camps, (themes.decision == "Fix-first alert").any(), start)
    R["evaluation"] = evaluate(R, posts)
    say("Done")
    return R


# --------------------------------------------------------------------------------------------
# Step 14: evaluation (only where ground truth or labelled data exists)
# --------------------------------------------------------------------------------------------
def evaluate(R, posts) -> pd.DataFrame:
    rows = []
    if "gt_dup_of" in posts:
        copies = set(posts.loc[posts.gt_dup_of.notna() & (posts.gt_dup_of != posts.post_id), "post_id"])
        removed = set(R["removed_map"])
        if removed and copies:
            rows.append(("Near-duplicate removal (MinHash + LSH)", "Precision / recall",
                         f"{len(removed & copies) / len(removed):.2f} / {len(removed & copies) / len(copies):.2f}"))
    sc = R["scored"]
    if "gt_relevant" in sc and sc["gt_relevant"].notna().all():
        p10 = [sc.nlargest(10, f"score::{k}")["gt_relevant"].mean() for k in R["queries"]]
        rows.append(("Relevance ranking (TF-IDF + cosine)", "Mean Precision@10", f"{np.mean(p10):.2f}"))
        kept = sc.relevance >= R["settings"]["relevance_threshold"]
        rows.append(("Relevance filter", "Precision / recall",
                     f"{sc.loc[kept, 'gt_relevant'].mean():.2f} / {sc.loc[kept, 'gt_relevant'].sum() / sc.gt_relevant.sum():.2f}"))
    rel = R["relevant"]
    t = R["topics"]
    purity = ""
    if "gt_theme" in rel and rel["gt_theme"].notna().all():
        purity = f" / purity {rel.groupby('topic')['gt_theme'].agg(lambda s: s.value_counts().iloc[0]).sum() / len(rel):.2f}"
    rows.append((f"Topic modeling ({t['method']})", f"NPMI coherence at k={t['k']}" + (" / purity" if purity else ""),
                 f"{t['coherence']:.3f}{purity}"))
    if R.get("sentiment_eval") is not None:
        for r in R["sentiment_eval"].itertuples():
            rows.append((f"Sentiment – {r.model}", "Accuracy / macro-F1 (held-out labelled posts)",
                         f"{r.accuracy:.2f} / {r.macro_f1:.2f}"))
    rs = R.get("reviews_scored")
    if rs is not None and "gt_fake" in rs:
        if rs.fake_flag.any():
            rows.append(("Fake-review alert", "Precision / recall",
                         f"{rs.loc[rs.fake_flag, 'gt_fake'].mean():.2f} / {rs.loc[rs.gt_fake, 'fake_flag'].mean():.2f}"))
        if rs.greenwash_flag.any():
            rows.append(("Greenwashing flag", "Precision / recall",
                         f"{rs.loc[rs.greenwash_flag, 'gt_greenwash'].mean():.2f} / {rs.loc[rs.gt_greenwash, 'greenwash_flag'].mean():.2f}"))
    th = R["themes"]
    rows.append(("Stock filter", "Themes promoted / held back",
                 f"{(th.decision == 'Promote').sum()} / {(th.decision != 'Promote').sum()}"))
    return pd.DataFrame(rows, columns=["Module", "Metric", "Result"])
