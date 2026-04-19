# Fashion Review Analytics: Market Basket Analysis, Text Mining & Predictive Modeling

**BAN 5743 — Predictive Analytics | Module 6**
**Dataset:** Amazon Reviews 2023 (Fashion vertical) · UCSD McAuley Lab
**Stack:** Python · mlxtend · scikit-learn · VADER · LDA · pandas

---

## Why This Project Exists

Retailers have a data abundance problem. Amazon's Fashion category generates millions of reviews, each one a dense signal about what customers bought, how they feel, and what they pair together. The challenge isn't accessing that data — it's knowing what questions to ask it simultaneously.

This project explores three analytical lenses applied to the same dataset, then asks what each lens can see that the others can't. Market basket analysis surfaces co-purchase structure. Text analytics surfaces semantic meaning. Predictive modeling operationalizes both into a classifier that can flag reviews before they damage a product's reputation. The real analytical work isn't in running any one of these techniques — it's in understanding where they agree, where they diverge, and why that divergence matters for actual business decisions.

---

## The Dataset and Its Honest Limitations

The data comes from Amazon's 2023 review corpus for the Fashion category: **78,351 reviews** from **8,347 unique users** covering **60,179 products**. Review dates span 2004 to 2023, though the bulk of activity clusters in 2014–2023.

The core limitation has to be stated plainly: **these are reviews, not purchase records.** We don't have shopping carts. We don't have checkout timestamps. We have the date a user wrote a review — which can lag the actual purchase by days, weeks, or sometimes months. Any co-purchase inference drawn from this data is a proxy. That proxy is meaningful, but it's not ground truth.

A second structural constraint: this dataset is extraordinarily sparse. 60,000 products across 8,000 users means most products are reviewed by a handful of people. The median user reviewed just 7 unique products across their entire Amazon history. This sparsity shapes every analytical decision downstream.

---

## Part 1 — Market Basket Analysis: The Basket Construction Problem

### The Core Decision: What Is a "Basket"?

This is where most analyses go wrong before they even start. A naive approach would treat each review as its own basket — one row, one item. That's not a basket; that's a product catalog. Association rules require co-occurrence within a transaction unit. The question is: what counts as one transaction for a fashion customer on Amazon?

The answer requires a judgment call. **We defined a basket as all unique products a single user reviewed within a 12-month calendar year.** Here's the reasoning:

Fashion purchasing is seasonal. Spring and fall are wardrobe refresh cycles. Holiday season drives gift purchases. Black Friday collapses cross-category buying into a short window. A one-year window is wide enough to capture these seasonal clusters while being narrow enough to exclude a customer's 2017 purchases from contaminating their 2023 basket. A shorter window (say, 90 days) would fragment too many purchase cycles that naturally span a season. A longer window (3+ years) would blur context entirely — a customer who bought a raincoat in 2019 and a swimsuit in 2022 didn't "co-purchase" those items in any meaningful sense.

### The Sparsity Problem Forced Calibration

With the chosen basket definition, we generated **27,059 annual user-baskets**, of which 16,029 contained two or more products. So far, so good.

The real problem emerged in the product frequency distribution. Across all baskets, the single most-reviewed product appeared in only **45 baskets**. The standard Apriori support threshold of 2% would require a product to appear in ~320 baskets — an impossible bar in this dataset. Setting it at 1% still requires 160 appearances. The data simply cannot produce those numbers.

This isn't a flaw in the data. It reflects how fashion works: niche products attract loyal but small audiences. A sustainable bamboo bracelet brand might have 30 devoted fans who reviewed multiple products in the same year. Those 30 buyers represent a real, coherent market signal — they're just not visible at thresholds calibrated for grocery retail.

The solution was to set the minimum support at **0.3%** (~4–5 basket appearances) and to filter the product universe to items appearing in at least 5 distinct baskets. After filtering, the transaction matrix shrank to **1,460 baskets × 869 products** — manageable for Apriori, and still containing real signal.

### What the Rules Actually Found

The algorithm produced **42 rules** across the filtered product set, with **10 strong rules** meeting both lift > 10 and confidence > 0.25. The top rule achieved a lift of **173.8** — meaning customers who reviewed product A were 174 times more likely to also review product B than random chance would predict.

Lift values this extreme typically indicate one of two things: either a brand-specific product line (different colorways or sizes of the same item sold as separate ASINs), or a genuine complementary pairing that a specific customer segment favors. Both are actionable. The brand-line case informs bundled listing strategy. The genuine pairing case informs cross-category recommendation placement.

The meaningful constraint is confidence. Rules with lift > 100 but confidence of 0.05 mean that even though the co-occurrence is vastly non-random, the majority of buyers of the antecedent item did *not* buy the consequent. That's not a recommendation engine rule — it's a niche curiosity. We prioritized rules where confidence exceeded 0.25, meaning at least 1 in 4 buyers of the antecedent item also appeared in a basket with the consequent.

---

## Part 2 — Text Analytics: Two Methods for Different Questions

### Why TF-IDF and VADER Instead of Just One

The temptation with text analysis is to pick a technique and commit to it. TF-IDF gives you words. Sentiment gives you tone. Topic modeling gives you themes. In isolation, each is incomplete.

**TF-IDF** was chosen to answer: *which words statistically distinguish negative reviews from positive ones?* It's discriminative by design — high TF-IDF weight means a term appears often in one rating group and rarely elsewhere. The results were stark and interpretable: 1-star reviews are dominated by *cheap, small, material, returned*, while 5-star reviews cluster around *cute, beautiful, perfect, comfortable*. These aren't surprising findings, but they're confirmable ones — the technique is doing what it's supposed to do.

**VADER sentiment** was chosen to answer a different question: *does a review's emotional tone correlate with its star rating, and by how much?* The answer is: yes, substantially but imperfectly. The Pearson correlation between VADER compound score and star rating is **0.55**. That's a meaningful relationship — positive reviews do skew positive in sentiment — but the 45% unexplained variance is where it gets interesting. Some 5-star reviews are written in muted, technical language that VADER scores neutrally. Some 2-star reviews express disappointment through complex sarcasm that VADER misses entirely. This gap between sentiment score and star rating is not noise; it's information. It's what the model in Part 3 has to work with.

### The Preprocessing Tradeoff

A deliberate choice was made to apply preprocessing (HTML stripping, stopword removal, lowercase normalization) to the TF-IDF features but **not to the VADER inputs**. VADER was built to handle natural language including punctuation, capitalization, and intensifiers like "TERRIBLE" vs "terrible." Stripping those signals before running VADER would systematically reduce its accuracy. TF-IDF, by contrast, benefits from clean tokenization — function words and HTML artifacts add noise without adding discriminative power.

### What the LDA Topics Revealed

Running LDA with 6 topics across the full corpus produced thematically coherent groupings that map onto recognizable fashion commerce patterns: quality and fit complaints, positive comfort and style language, jewelry and accessories discourse, shipping and packaging mentions, material and fabric focus, and gift/occasion purchasing. The distribution of these topics across star ratings confirmed the expected pattern — quality complaint topics are over-represented in 1–2 star reviews, positive comfort language dominates 4–5 star reviews — but the interesting finding was topic overlap. Quality complaint language appears even in 4-star reviews, just at lower frequency. Customers who are mostly satisfied still document minor fit issues. This has implications for how you'd use topic features in a classifier: topic presence alone isn't a clean signal; topic weight relative to other topics matters more.

---

## Part 3 — Predictive Modeling: The Setup and the Choices

### Task Definition

The prediction task is binary classification: **positive (≥4 stars) vs. negative (≤2 stars)**. Three-star reviews are excluded. This is deliberate — neutral reviews are linguistically ambiguous and often represent fundamentally different customer mindsets (disappointed but forgiving, vs. genuinely neutral). Including them would contaminate both classes without adding useful signal.

The class imbalance is significant: **85.2% of the binary-eligible reviews are positive**. A naive classifier predicting "positive" for everything would achieve 85% accuracy. That's why accuracy is not reported as the primary metric.

### Feature Engineering

The feature matrix combines three sources:
- **TF-IDF** (5,000 features, unigrams + bigrams, sublinear TF, min_df=5) — fitted on training data only to prevent leakage
- **VADER scores** (compound, positive, negative, neutral proportions) — computed on raw text, not preprocessed text
- **Metadata** (text length in characters, verified purchase flag)

The decision to include text length as a feature reflects an observed pattern: negative reviews tend to be longer. Customers who feel wronged write more. Verified purchase status is included because unverified reviews have a different distribution of sentiment — they skew more extreme in both directions.

**Critical detail:** TF-IDF was fit on the training split only. Fitting on the full dataset before splitting would allow the vectorizer to "see" the test set vocabulary, inflating performance estimates. With 68,878 samples split 80/20, this isn't a catastrophic issue, but it's a correctness principle worth maintaining.

### Model Selection

Two models were trained:

**Logistic Regression** (L2, balanced class weights, saga solver)
AUC = 0.858 · F1 = 0.927

**Random Forest** (200 trees, max_depth=12, balanced weights)
AUC = 0.938 · F1 = 0.931

Random Forest achieves a notably higher AUC (0.938 vs 0.858), suggesting it captures non-linear interactions between text features that Logistic Regression misses. However, **Logistic Regression is selected as the primary model** for two reasons:

1. The F1 scores are nearly identical (0.927 vs 0.931). For deployment purposes, the 0.004 F1 gap doesn't justify the interpretability cost.
2. Logistic Regression coefficients directly answer the question "which specific words or features drive this prediction?" Random Forest feature importances tell you what mattered, but not in which direction or by how much.

In a business context, explaining to a category manager *why* a review is flagged as negative requires knowing that "returned cheap nothing like" is the driver — not that "feature #3847 had importance 0.002."

### What the Coefficients Revealed

The most positive-predicting terms: *perfect, beautiful, recommend, exactly, comfortable, adorable, quality* — language of met or exceeded expectations.

The most negative-predicting terms: *returned, terrible, disappointed, cheap, nothing like, poor quality* — language of product-reality mismatch and quality failure.

The VADER compound score itself emerged as a highly influential feature, sitting in the top 20 predictors for Logistic Regression. This validates including it explicitly: even after the model has access to the raw word features, knowing the overall sentiment polarity adds independent predictive value. Words carry context that VADER aggregates in ways that TF-IDF's bag-of-words representation doesn't fully capture.

---

## Integration Findings: Where the Methods Agree and Disagree

The most analytically valuable finding wasn't in any single method — it was in comparing them.

**Where they agree:** Products that appear in high-lift association rules tend to have higher average sentiment scores and ratings than the general product population. This makes intuitive sense: customers who are happy with a product are more likely to buy related items from a similar brand or style category. MBA and text analytics point in the same direction here.

**Where they diverge:** Two products can co-occur at lift > 40 while having meaningfully different average sentiment profiles and review language. This divergence suggests the co-purchase is driven by *functional complementarity* — the items are used together — rather than *semantic similarity* — customers feel the same way about both. A dress and a belt that customers consistently buy together may generate completely different reviews. The dress might attract complaints about sizing; the belt might attract praise for quality. MBA sees them as a pair. Text analytics sees them as distinct customer experiences. Neither is wrong.

This distinction matters for bundling decisions. A bundle of two positively-reviewed, co-occurring products is a safe cross-sell. A bundle where one item has negative sentiment is a risk — you're potentially associating your recommended product with a customer's dissatisfaction.

---

## Repository Structure

```
.
├── Module6.ipynb               # Main analysis notebook (all three parts + reflection)
├── Fashion_Reviews.csv         # Review data (78K rows)
├── Fashion_products.csv        # Product metadata
└── README.md                   # This file
```

Generated figures (saved during notebook execution):
- `fig_data_overview.png` — Rating distribution and reviews per year
- `fig_association_rules.png` — Support/confidence scatter + top rules by lift
- `fig_tfidf_terms.png` — Top TF-IDF terms for 1★ vs 5★ reviews
- `fig_sentiment.png` — VADER compound distribution by rating
- `fig_topic_by_rating.png` — LDA topic mix across star ratings
- `fig_model_evaluation.png` — ROC curves + confusion matrices
- `fig_feature_importance.png` — Logistic Regression coefficients
- `fig_rf_importance.png` — Random Forest feature importances

---

## Setup & Reproduction

```bash
# Python 3.9+ required
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend vaderSentiment

# Run the notebook
jupyter notebook Module6.ipynb
```

No NLTK downloads required — stopwords are bundled directly in the notebook to ensure the analysis runs in network-restricted environments.

Data files must be in the same directory as the notebook. The CSVs are not included in this repository due to size; they originate from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/main.html) (Fashion subset).

---

## Key Methodological Decisions at a Glance

| Decision | Choice | Why |
|---|---|---|
| Basket window | 12-month calendar year | Seasonal alignment; avoids conflating multi-year context |
| Min product frequency | 5 baskets | Fashion sparsity makes higher thresholds impractical |
| Apriori support threshold | 0.3% | Calibrated to max observed product frequency (~45 baskets) |
| Strong rule threshold | Lift > 10, Confidence > 0.25 | Filters trivial lift while requiring meaningful directionality |
| Sentiment preprocessing | Raw text only (no stopword removal) | Preserves VADER's context sensitivity |
| Neutral reviews | Excluded from classifier | Reduces label noise in binary task |
| Primary model | Logistic Regression | Interpretability outweighs marginal AUC gain from RF |
| Evaluation metrics | AUC-ROC + F1 | Handles class imbalance; accuracy alone misleads at 85% positive rate |

---

## Limitations and Future Work

**Data proxy:** Review-based baskets are a proxy for purchase behavior. Customers who review multiple items in a year may have bought them at different times and for different reasons. A better dataset would include actual cart or order data.

**Product identity:** ASIN-level co-occurrence may conflate related products (different sizes/colors of the same item) with genuinely complementary product pairs. Grouping by brand or product category before running MBA could reduce this confound.

**Temporal drift:** The dataset spans nearly 20 years (2004–2023). Customer language, fashion trends, and product categories have shifted substantially over that period. Models trained on historical reviews may not generalize to current review patterns without temporal validation.

**Sentiment model calibration:** VADER was designed for social media text and performs well on casual review language, but it was not trained on fashion-specific vocabulary. Domain-adapted sentiment models (or fine-tuned transformers on fashion reviews) would likely improve both the correlation with star ratings and the predictive utility of sentiment features.

---

*Analysis conducted as part of BAN 5743 Predictive Analytics coursework. Dataset sourced from the Amazon Reviews 2023 project, maintained by the UCSD McAuley Lab.*
