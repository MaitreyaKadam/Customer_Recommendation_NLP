# Predicting Customer Recommendations from Review Text
### An NLP pipeline for e-commerce product recommendation classification

## Business Problem

Online apparel retailers rely heavily on customer reviews to guide shoppers toward the right products but manually reading thousands of reviews to gauge sentiment doesn't scale. A key signal retailers want to predict automatically is: **will a customer recommend this product to others, based on what they wrote?**

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
- **Does adding the review title (not just the body) improve prediction accuracy?** i.e., is it worth the extra engineering effort to ingest more fields from the review form?

## Data

The underlying dataset is a modified version of the [Women's E-Commerce Clothing Reviews dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (~19,600 reviews), using the review title, review body text, and a binary recommendation label as the core fields. Raw data files are excluded from this repo; see `docs/results_summary.md` for a description of the schema.

## Key Results

## Model: Logistic Regression: Does adding different features help?

*For Title*

| Feature Representation | Mean Accuracy | Mean Weighted F1 |
|---|---|---|
| Bag-of-Words (CountVectorizer) | 0.8850 | 0.8789 |
| Unweighted FastText Embeddings | 0.8173 | 0.7363 | 
| TF-IDF Weighted FastText Embeddings | 0.8175 | 0.7491 | 

*For Review Text(Description)*

| Feature Representation | Mean Accuracy | Mean Weighted F1 |
|---|---|---|
| Bag-of-Words (CountVectorizer) | 0.8753 | 0.8686 |
| Unweighted FastText Embeddings | 0.8397 | 0.8133 | 
| TF-IDF Weighted FastText Embeddings | 0.8489 | 0.8287 | 

*For Title & Review Text*

| Feature Representation | Mean Accuracy | Mean Weighted F1 |
|---|---|---|
| Bag-of-Words (CountVectorizer) | 0.9008 | 0.8980 |
| Unweighted FastText Embeddings | 0.8516 | 0.8311 | 
| TF-IDF Weighted FastText Embeddings | 0.8535 | 0.8370 | 


From the above table, it is very clear that combining title and review description has improved the performance of the logistic regression model. The model achieved the highest accuracy of 0.9008 and F1-score of 0.8980 with the combined Bag-of-words features. This score is greater than the score obtained for just the title and just the review description. This suggests that using word counts from both the title and the review description provides the most effective information for prediction recommnedations. The unweighted and TF-IDF weighted embeddings also showed improvement when combined, but they still underperformed compared to the Bag-of-Words features.

## Tech Stack
Python · pandas · scikit-learn · NLTK / regex tokenization · gensim (word embeddings) · Jupyter

## What I'd Do With More Time / In Production
- Swap logistic regression for gradient-boosted trees or a fine-tuned transformer (e.g. DistilBERT) and compare lift over the classical baselines.
- Handle class imbalance explicitly (recommendation labels are rarely 50/50 in review data) with class weighting or resampling.
- Serve the best model behind a lightweight API and wire it into a product page as a "predicted satisfaction" signal.


