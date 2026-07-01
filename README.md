# Predicting Customer Recommendations from Review Text
### An NLP pipeline for e-commerce product recommendation classification

## Business Problem

Online apparel retailers rely heavily on customer reviews to guide shoppers toward the right products — but manually reading thousands of reviews to gauge sentiment doesn't scale. A key signal retailers want to predict automatically is: **will a customer recommend this product to others, based on what they wrote?**

This matters commercially because:
- **Product ranking & merchandising** — items with a high predicted recommendation rate can be surfaced higher in search/browse results.
- **Early warning system** — a spike in reviews with low predicted recommendation scores can flag a quality or sizing issue before it shows up in return rates.
- **Recommendation engines** — the classifier output becomes a feature for downstream "customers also loved" logic.

This project builds the natural language processing pipeline that powers that prediction: taking raw, unstructured review text and turning it into a model that classifies whether a review represents a "recommend" or "not recommend" outcome.

## Approach

The pipeline is split into three stages:

**1. Text Cleaning & Vocabulary Construction**
Raw review text is tokenized, normalized, and filtered (stopwords, rare/overly common terms) to build a clean, reusable vocabulary — the foundation any downstream model depends on.

**2. Feature Engineering**
Three different ways of representing review text numerically are built and compared:
- A **Bag-of-Words / count-vector** representation (interpretable, fast baseline)
- **Unweighted word embeddings** (captures semantic meaning, e.g. "great" and "excellent" are close together)
- **TF-IDF weighted embeddings** (semantic meaning + emphasis on distinctive words per review)

**3. Classification & Evaluation**
Logistic regression models are trained on each representation and evaluated with 5-fold cross-validation to answer two business-relevant questions:
- **Which text representation predicts recommendation outcomes most accurately?**
- **Does adding the review title (not just the body) improve prediction accuracy?** — i.e., is it worth the extra engineering effort to ingest more fields from the review form?


## Data

The underlying dataset is a modified version of the [Women's E-Commerce Clothing Reviews dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (~19,600 reviews), using the review title, review body text, and a binary recommendation label as the core fields. Raw data files are excluded from this repo; see `docs/results_summary.md` for a description of the schema.

## Key Results

*(fill in once you've pulled your actual numbers — see prompts below)*

| Feature Representation | Model | Cross-Val Accuracy |
|---|---|---|
| Bag-of-Words (count vectors) | Logistic Regression | — |
| Unweighted embeddings | Logistic Regression | — |
| TF-IDF weighted embeddings | Logistic Regression | — |

**Does more text (title + body) help?**
*(one or two sentences summarizing the title-only vs. body-only vs. combined comparison)*

## Tech Stack
Python · pandas · scikit-learn · NLTK / regex tokenization · gensim (word embeddings) · Jupyter

## What I'd Do With More Time / In Production
- Swap logistic regression for gradient-boosted trees or a fine-tuned transformer (e.g. DistilBERT) and compare lift over the classical baselines.
- Handle class imbalance explicitly (recommendation labels are rarely 50/50 in review data) with class weighting or resampling.
- Serve the best model behind a lightweight API and wire it into a product page as a "predicted satisfaction" signal.

---
*This project was completed as coursework and is presented here as a portfolio piece demonstrating an end-to-end NLP feature engineering and classification workflow.*
