# Task 4: Sentiment Analysis — Product Review Classifier

## Topic
Analyze product review text and classify it as **Positive**, **Negative**,
or **Neutral** using NLP lexicon-based sentiment scoring (VADER).

## Folder Structure
```
task4-sentiment-analysis/
├── generate_dataset.py     -> Creates a realistic sample reviews dataset
├── analyze_sentiment.py    -> Runs VADER sentiment analysis + builds charts
├── data/
│   └── product_reviews.csv      (300 sample reviews across 5 products)
└── output/
    ├── sentiment_distribution.png
    ├── sentiment_by_product.png
    ├── score_distribution.png
    ├── rating_vs_sentiment.png
    └── reviews_with_sentiment.csv   (full dataset + predicted sentiment)
```

## Why VADER?
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon
and rule-based sentiment tool built specifically for short, informal text —
exactly what you find in product reviews, tweets, and social media posts.
It understands intensifiers ("very good" vs "good"), punctuation ("great!!!"),
capitalization ("AMAZING"), and negation ("not good").

Each review gets a **compound score** from -1 (very negative) to +1 (very
positive), which we bucket into:
- `compound >= 0.05` → Positive
- `compound <= -0.05` → Negative
- otherwise → Neutral

## Dataset
`product_reviews.csv` has 300 sample reviews across 5 products (Wireless
Earbuds, Smartwatch, Air Fryer, Running Shoes, Backpack) with columns:
`review_id, product, review_date, rating, review_text, true_label`.

> In a real project, replace this with scraped Amazon/Flipkart reviews,
> tweets pulled via the Twitter/X API, or app store reviews — just keep a
> `review_text` column and the same script will work unchanged.

## How to Run

### 1. Install dependencies (one time)
```bash
pip install pandas matplotlib seaborn nltk
```

### 2. Generate the sample dataset
```bash
cd task4-sentiment-analysis
python generate_dataset.py
```

### 3. Run sentiment analysis
```bash
python analyze_sentiment.py
```
First run downloads the VADER lexicon automatically (one-time, needs
internet). Prints a summary in the terminal and saves charts + a labeled
CSV to `output/`.

## What the Charts Show
| Chart | Insight |
|---|---|
| `sentiment_distribution.png` | Overall split: % positive / negative / neutral |
| `sentiment_by_product.png` | Which products get the most complaints vs praise |
| `score_distribution.png` | How strongly positive/negative reviews skew |
| `rating_vs_sentiment.png` | Whether star ratings match the text sentiment |

## Real-World Uses
- **Marketing**: spot which products need a messaging fix
- **Product development**: surface recurring complaints automatically
- **Social listening**: track public opinion shifts over time
