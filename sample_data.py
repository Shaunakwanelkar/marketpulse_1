"""Synthetic sample data for MarketPulse (clearly labelled as fictional).

Generates X-style posts, an inventory sheet, customer reviews, weekly sales and a small
hand-labelled post set. Every record also carries hidden ground-truth columns (gt_*) that the
app uses ONLY to score its own methods on the Evaluation tab.
"""
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

SAMPLE_KEYWORDS = """eco diwali: diya candle soy wax gift festive green sustainable
sustainable gifts: gift wrap fabric card seed hamper reusable eco
plastic-free: plastic free zero waste reusable bamboo steel compostable biodegradable
EcoNest: econest brand order delivery package parcel customer care"""


def generate_sample(seed: int = 42) -> dict:
    random.seed(seed); np.random.seed(seed)
    # ---------- Keywords and inventory ----------

    inventory = pd.DataFrame([
        # sku, product, category, channels, stock_units, eco_claims, tags (words that link a theme to this product)
        ("P01", "Reusable Fabric Gift Wrap",          "Gifting",       "Amazon; Sustainable marketplace",          400, "reusable; upcycled cotton", "gift wrap wrapping furoshiki"),
        ("P02", "Bamboo Toothbrush (Pack of 4)",       "Personal Care", "Amazon; Blinkit; Sustainable marketplace", 900, "biodegradable; plastic-free", "toothbrush bamboo bristle brush"),
        ("P03", "Steel Water Bottle 750ml",            "Kitchen",       "Amazon; Blinkit",                           30, "reusable", "bottle steel water"),
        ("P04", "Soy Wax Diya Candles (Set of 6)",     "Home",          "Sustainable marketplace; Amazon",          250, "soy wax; plastic-free", "diya candle soy wax lamp"),
        ("P05", "Plantable Seed Paper Greeting Cards", "Gifting",       "Sustainable marketplace",                  600, "plantable; recycled paper", "card seed paper plant greeting"),
        ("P06", "Coconut Coir Dish Scrubber",          "Kitchen",       "Blinkit; Amazon",                         1200, "compostable; plastic-free", "scrubber coir dish"),
        ("P07", "Organic Cotton Tote Bag",             "Fashion",       "Sustainable marketplace",                    0, "organic cotton", "tote bag"),
        ("P08", "Eco Diwali Gift Hamper",              "Gifting",       "Sustainable marketplace; Amazon",           80, "plastic-free; handmade", "hamper"),
    ], columns=["sku", "product", "category", "channels", "stock_units", "eco_claims", "tags"])

    # ---------- Synthetic X posts ----------
    # Each theme has sentence pools by polarity. A post = 1–2 sentences of one polarity + optional filler + noise.
    THEMES = {
      "eco_gift_wrap": dict(base=16, growth=0.12, pol=(0.70, 0.20, 0.10), pos=[
          "Switched to reusable fabric gift wrap this year and it looks so festive",
          "Furoshiki style cloth wrapping is the best idea for diwali gifts",
          "No more shiny wrapping paper, cloth gift wrap all the way",
          "Wrapped every diwali gift in fabric and my family loved it",
          "Reusable gift wrap is beautiful and we can use it again next year",
          "Cloth wrap made our gift boxes look premium and zero waste",
          "Loving how fabric gift wrapping cuts down festive waste",
          "Gift wrapping with old sarees and fabric wraps feels so special"],
        neu=[
          "Where can I buy reusable gift wrap before diwali",
          "Is fabric gift wrap cheaper than paper wrapping in the long run",
          "Thinking of trying cloth wrapping for gifts this festive season",
          "Any tips for furoshiki style gift wrapping"],
        neg=[
          "Fabric gift wrap is too expensive for wrapping twenty gifts",
          "Tried cloth wrapping and the knots kept coming loose on every gift",
          "Reusable gift wrap arrived creased and looked cheap"]),
      "diwali_candles": dict(base=11, growth=0.12, pol=(0.65, 0.25, 0.10), pos=[
          "Soy wax diyas burn so clean, no black smoke at home",
          "Bought soy wax candles for diwali and the glow is lovely",
          "Clay diyas with soy wax are my favourite eco diwali find",
          "These soy candles last all evening and smell natural",
          "Lit soy wax diyas tonight and the house feels so warm",
          "Eco diwali lights done right with soy wax diya sets"],
        neu=[
          "Are soy wax diyas better than paraffin candles",
          "Need diya candle recommendations for diwali night",
          "Comparing soy wax and beeswax candles for the festival"],
        neg=[
          "Soy wax diyas melted too fast and the wick kept drowning",
          "Paid extra for soy candles and they barely lasted an hour"]),
      "steel_bottle": dict(base=14, growth=0.0, pol=(0.55, 0.30, 0.15), pos=[
          "My steel water bottle keeps water cold the whole day at office",
          "Ditched plastic bottles, steel bottle is the best switch",
          "Love my stainless steel bottle, no more buying plastic water",
          "Steel bottle survived a fall and still looks good",
          "Carrying a steel water bottle to college every day now"],
        neu=[
          "Which steel water bottle size is good for office use",
          "Is a 750 ml steel bottle enough for a day",
          "Steel or copper water bottle for daily use"],
        neg=[
          "My steel bottle lid started leaking after two weeks",
          "Steel water bottle has a metallic taste, disappointed",
          "Bottle cap leaks in my bag every single time"]),
      "bamboo_toothbrush": dict(base=13, growth=0.0, pol=(0.40, 0.25, 0.35), pos=[
          "Bamboo toothbrush feels great and no plastic handle",
          "Switched the whole family to bamboo toothbrushes",
          "Bamboo brush bristles are soft and the handle feels nice",
          "Happy with my bamboo toothbrush, easy plastic free swap"],
        neu=[
          "How often should you change a bamboo toothbrush",
          "Can you compost a bamboo toothbrush handle",
          "Bamboo toothbrush or electric brush, which is better"],
        neg=[
          "Great, another bamboo toothbrush that sheds bristles in a week",
          "Bamboo toothbrush handle got mouldy after a month",
          "Bristles on my bamboo brush fell out, waste of money",
          "Bamboo toothbrush bristles are too hard and hurt my gums"]),
      "seed_paper_cards": dict(base=6, growth=0.10, pol=(0.70, 0.25, 0.05), pos=[
          "Sent plantable seed paper cards for diwali and friends planted them",
          "Seed paper greeting cards that grow into tulsi, so cute",
          "Our office used seed paper cards this year, lovely idea",
          "Plantable cards are the sweetest eco friendly greeting"],
        neu=[
          "Where to buy seed paper greeting cards in bulk",
          "Do seed paper cards actually grow plants"],
        neg=[
          "Planted the seed paper card but nothing grew"]),
      "delivery_problems": dict(base=5, growth=0.0, late_boost=18, pol=(0.05, 0.15, 0.80), pos=[
          "Quick commerce delivered my diwali order in ten minutes, impressed"],
        neu=[
          "Is there a delivery delay during diwali week",
          "Tracking says out for delivery since morning"],
        neg=[
          "Diwali order delivery is late again, still waiting for my gifts",
          "Package arrived damaged and the box was crushed",
          "Delivery got cancelled last minute before diwali, so frustrating",
          "Ordered eco gifts a week ago and delivery keeps getting delayed",
          "Late delivery ruined my diwali gifting plan",
          "Received a damaged parcel and no reply from customer care"]),
      "greenwashing": dict(base=7, growth=0.0, pol=(0.05, 0.25, 0.70), pos=[
          "Glad some brands share real certificates for their eco claims"],
        neu=[
          "How do you check if a product is really compostable",
          "What does biodegradable actually mean on packaging"],
        neg=[
          "Brand says plastic free but the product came wrapped in plastic",
          "So called eco friendly products are just greenwashing",
          "Claims compostable but it never broke down in my compost bin",
          "Tired of fake sustainable labels on everything this diwali",
          "Not really biodegradable if it needs an industrial facility"]),
      "plastic_free_general": dict(base=13, growth=0.0, pol=(0.40, 0.50, 0.10), pos=[
          "Going plastic free at home one small swap at a time",
          "Our plastic free kitchen journey is going well",
          "Zero waste living is easier than I thought"],
        neu=[
          "Looking for plastic free alternatives for daily use",
          "What are easy zero waste swaps for beginners",
          "Plastic free july habits I am keeping for the rest of the year",
          "Sustainable living tips for small apartments"],
        neg=[
          "Plastic free products are so hard to find in local shops",
          "Zero waste shopping is too expensive for most people"]),
    }
    OFFTOPIC = [  # mention diwali/festive words but are not about eco products (relevance negatives)
        "Diwali sweets shopping done, kaju katli overload", "Crackers traffic jam near the market tonight",
        "Watching the cricket match before diwali puja", "New movie releasing on diwali weekend, so excited",
        "Diwali bonus finally credited in my account", "Rangoli competition at our society tomorrow",
        "Diwali office party photos coming soon", "Gold prices are rising before diwali",
        "Train tickets for diwali are sold out again", "Made gulab jamun at home for the festival"]
    FILLER = {"positive": ["Highly recommend!", "Totally worth it.", "So happy with this.", "Love it."],
              "neutral": ["Any suggestions?", "Just wondering.", "Thoughts?", "Need advice."],
              "negative": ["Not happy.", "So disappointed.", "Never again.", "Really annoying."]}
    HASHTAGS = ["#EcoDiwali", "#SustainableLiving", "#PlasticFree", "#ZeroWaste", "#DiwaliGifts", "#GoGreen", "#SustainableGifts"]
    EMOJIS = ["✨", "🪔", "🌱", "🎁", "💚", "🙄", "😡", "😊", "♻️"]

    START = date(2026, 10, 20); DAYS = 14
    posts = []
    OPENERS = ["Honestly", "PSA:", "Okay so", "Day 3 of festive prep and", "Not gonna lie,", "Quick update:",
               "Mumbai folks,", "Small win today:", "Tell me why", "Real talk:", "Update from Pune:", "For the record,",
               "Weekend thought:", "Just saying,", "Hot take:", "Note to self:", "Family chat says", "Friendly reminder:",
               "Bengaluru check in:", "Delhi diaries:", "Morning thought:", "So apparently", "Story time:", "FYI"]
    CLOSERS = ["what do you all think", "more updates soon", "anyone else", "rate this out of ten", "lesson learnt",
               "sharing in case it helps", "will post pics later", "thread below", "that is all", "mood right now",
               "tell me I am not alone", "noted for next year", "adding to my list", "end of rant", "worth sharing"]

    def noise(text, pol):
        """Add the mess and variety real posts have: openers, mentions, URLs, emojis, hashtags, brand name."""
        if random.random() < 0.75: text = random.choice(OPENERS) + " " + text[0].lower() + text[1:]
        if random.random() < 0.55: text += ", " + random.choice(CLOSERS)
        if random.random() < 0.25: text = f"@{random.choice(['friend','sis','team','ravi_k','eco_mom'])} " + text
        if random.random() < 0.30: text += " " + random.choice(FILLER[pol])
        if random.random() < 0.15: text += " via EcoNest"
        if random.random() < 0.50: text += " " + " ".join(random.sample(HASHTAGS, k=random.randint(1, 3)))
        if random.random() < 0.40: text += " " + random.choice(EMOJIS)
        if random.random() < 0.20: text += f" https://t.co/{random.randrange(16**8):08x}"
        return text

    pid = 0
    for d in range(DAYS):
        day = START + timedelta(days=d)
        for theme, cfg in THEMES.items():
            n = cfg["base"] * (1 + cfg["growth"] * d) + (cfg.get("late_boost", 0) if d >= 9 else 0)
            for _ in range(np.random.poisson(n)):
                pol = np.random.choice(["positive", "neutral", "negative"], p=cfg["pol"])
                pool = cfg[{"positive": "pos", "neutral": "neu", "negative": "neg"}[pol]]
                k = 2 if (len(pool) > 2 and random.random() < 0.4) else 1
                text = ". ".join(random.sample(pool, k))
                posts.append(dict(post_id=f"x{pid:05d}", created_at=str(day), text=noise(text, pol),
                                  likes=int(np.random.gamma(1.2, 8)), gt_theme=theme, gt_sentiment=pol,
                                  gt_relevant=True, gt_dup_of=None)); pid += 1
        for _ in range(np.random.poisson(22)):
            text = random.choice(OFFTOPIC)
            posts.append(dict(post_id=f"x{pid:05d}", created_at=str(day), text=noise(text, "neutral"),
                              likes=int(np.random.gamma(1.2, 8)), gt_theme="off_topic", gt_sentiment="neutral",
                              gt_relevant=False, gt_dup_of=None)); pid += 1

    # Copy-paste spam campaigns (near-duplicates of each other)
    SPAM = ["Flat 70 percent off on all diwali gift hampers today only, order now",
            "Biggest festive sale is live, eco diwali gifts at lowest prices, grab the deal",
            "Win free diwali gifts, follow and retweet to enter the giveaway",
            "Best price on plastic free products this diwali, limited stock, buy now",
            "Sustainable gifts mega sale, extra 20 percent off with code DIWALI20"]
    for s_i, s in enumerate(SPAM):
        first = None
        for _ in range(random.randint(18, 35)):
            day = START + timedelta(days=random.randrange(DAYS))
            text = s + " " + random.choice(["🎁", "🔥", "!!", "✨", ""]) + f" https://bit.ly/{random.randrange(16**6):06x}"
            rec = dict(post_id=f"x{pid:05d}", created_at=str(day), text=text, likes=0, gt_theme="spam",
                       gt_sentiment="neutral", gt_relevant=False, gt_dup_of=first or f"x{pid:05d}")
            first = first or rec["post_id"]; posts.append(rec); pid += 1

    # Retweets of genuine posts (exact or near copies)
    genuine = [p for p in posts if p["gt_relevant"]]
    for orig in random.sample(genuine, 160):
        posts.append(dict(post_id=f"x{pid:05d}", created_at=orig["created_at"],
                          text=f"RT @{random.choice(['ecowarrior','greenindia','mumbai_mom','zerowaste_in'])}: " + orig["text"],
                          likes=0, gt_theme=orig["gt_theme"], gt_sentiment=orig["gt_sentiment"],
                          gt_relevant=True, gt_dup_of=orig["post_id"])); pid += 1

    posts = pd.DataFrame(posts).sample(frac=1, random_state=seed).reset_index(drop=True)

    # ---------- Synthetic customer reviews (12 weeks) ----------
    ASPECT_SENTS = {
      "quality":   {"positive": ["The {p} quality is excellent", "Really well made {p}", "Sturdy and well finished"],
                    "neutral":  ["The {p} is okay for the price", "Quality is average"],
                    "negative": ["Poor quality {p}, feels cheap", "The {p} broke within days"]},
      "packaging": {"positive": ["Lovely plastic free packaging", "Packed beautifully in paper"],
                    "neutral":  ["Packaging was simple"],
                    "negative": ["Packaging was torn when it arrived", "Too much packaging for one item"]},
      "delivery":  {"positive": ["Delivery was quick", "Arrived before the promised date"],
                    "neutral":  ["Delivery took the usual time"],
                    "negative": ["Delivery was very late", "Delivery took two weeks"]},
      "price":     {"positive": ["Great value for money", "Price is fair for an eco product"],
                    "neutral":  ["A bit pricey but fine"],
                    "negative": ["Way too expensive", "Overpriced for what you get"]},
      "durability":{"positive": ["Still going strong after months", "Lasts long"],
                    "neutral":  ["Durability remains to be seen"],
                    "negative": ["Stopped working after a few weeks", "Wore out quickly"]},
      "eco_claim": {"positive": ["Love that it is genuinely plastic free", "Happy it is truly compostable"],
                    "neutral":  ["Not sure how eco friendly it really is"],
                    "negative": ["Claims compostable but it never broke down", "Says plastic free but came wrapped in plastic",
                                 "Not really biodegradable as advertised"]},
    }
    PRODUCT_ISSUE = {"P02": ("durability", "The bristles started falling out"),
                     "P03": ("durability", "The lid leaks water")}
    STAR_DIST = {"P01": [.03,.04,.08,.30,.55], "P02": [.10,.12,.18,.30,.30], "P03": [.08,.10,.15,.32,.35],
                 "P04": [.03,.05,.10,.32,.50], "P05": [.04,.05,.11,.30,.50], "P06": [.05,.06,.14,.35,.40],
                 "P07": [.03,.05,.12,.35,.45], "P08": [.05,.07,.15,.33,.40]}
    REV_START = date(2026, 8, 10); WEEKS = 12
    short = {r.sku: r.product.split("(")[0].strip().lower() for r in inventory.itertuples()}
    reviews = []; rid = 0
    for sku in STAR_DIST:
        for w in range(WEEKS):
            festive = 1 + (0.15 * w if sku in ("P01", "P04", "P05", "P08") else 0)
            for _ in range(np.random.poisson(6 * festive)):
                stars = int(np.random.choice([1,2,3,4,5], p=STAR_DIST[sku]))
                pol_p = {5:(.9,.08,.02), 4:(.75,.2,.05), 3:(.3,.4,.3), 2:(.08,.22,.7), 1:(.03,.1,.87)}[stars]
                aspects = random.sample(list(ASPECT_SENTS), k=random.randint(2, 3))
                sents, green = [], False
                for a in aspects:
                    pol = np.random.choice(["positive","neutral","negative"], p=pol_p)
                    s = random.choice(ASPECT_SENTS[a][pol]).format(p=short[sku])
                    green |= (a == "eco_claim" and pol == "negative"); sents.append(s)
                if sku in PRODUCT_ISSUE and stars <= 3 and random.random() < 0.6:
                    sents.append(PRODUCT_ISSUE[sku][1])
                day = REV_START + timedelta(days=7*w + random.randrange(7))
                reviews.append(dict(review_id=f"r{rid:05d}", sku=sku, date=str(day), stars=stars,
                                    text=". ".join(sents) + ".", gt_fake=False, gt_greenwash=green)); rid += 1
    # A burst of near-identical 5-star reviews for the hamper on one day (fake-review pattern)
    for i in range(28):
        text = random.choice(["Best eco diwali hamper ever!!! Must buy, amazing quality and packaging.",
                              "Best eco diwali hamper ever!! Must buy, amazing quality and packaging",
                              "Best eco diwali hamper ever! Must buy amazing quality and packaging."])
        reviews.append(dict(review_id=f"r{rid:05d}", sku="P08", date="2026-10-15", stars=5, text=text,
                            gt_fake=True, gt_greenwash=False)); rid += 1
    reviews = pd.DataFrame(reviews)

    # ---------- Synthetic weekly sales ----------
    # Units for some products respond to positive reviews that week; others do not (so correlations differ).
    rv = reviews[~reviews.gt_fake].assign(week=lambda d: (pd.to_datetime(d.date) - pd.Timestamp(REV_START)).dt.days // 7)
    pos_week = rv[rv.stars >= 4].groupby(["sku", "week"]).size()
    RESP = {"P01": 6, "P04": 5, "P05": 4, "P08": 5, "P03": 2, "P02": 0, "P06": 0, "P07": 0}
    BASE = {"P01": 25, "P02": 80, "P03": 40, "P04": 20, "P05": 15, "P06": 60, "P07": 18, "P08": 10}
    sales = []
    for sku in RESP:
        for w in range(WEEKS):
            units = BASE[sku] + RESP[sku] * pos_week.get((sku, w), 0) + np.random.normal(0, 8)
            sales.append(dict(sku=sku, week=w, week_start=str(REV_START + timedelta(days=7*w)), units=max(0, int(units))))
    sales = pd.DataFrame(sales)

    # A "hand-labelled" sample of posts (in real use you would label ~500 posts yourself)
    labelled = posts[posts.gt_relevant & posts.gt_dup_of.isna()].sample(700, random_state=seed)
    labelled = labelled.rename(columns={"gt_sentiment": "label"})[["text", "label"]].reset_index(drop=True)
    return dict(inventory=inventory, posts=posts, reviews=reviews, sales=sales, labelled=labelled)
