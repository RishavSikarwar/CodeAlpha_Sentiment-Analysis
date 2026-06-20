"""
TASK 4: SENTIMENT ANALYSIS
============================
Analyzes product review text and classifies each review as
Positive, Negative, or Neutral using NLP lexicon-based scoring
(VADER - Valence Aware Dictionary and sEntiment Reasoner).

VADER is well-suited for short, informal text like reviews, tweets,
and social media posts (it understands punctuation, emojis, capitalization,
and intensifiers like "very" or "absolutely").

Run:
    python analyze_sentiment.py
(Run generate_dataset.py first if data/product_reviews.csv doesn't exist)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "product_reviews.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure the VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
sns.set_theme(style="whitegrid")


def classify(text: str) -> tuple[str, float]:
    """Returns (label, compound_score) for a piece of text."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound


def load_and_score() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    results = df["review_text"].apply(classify)
    df["predicted_label"] = results.apply(lambda x: x[0])
    df["sentiment_score"] = results.apply(lambda x: x[1])
    return df


def print_summary(df: pd.DataFrame):
    print("\n--- SENTIMENT ANALYSIS SUMMARY ---")
    counts = df["predicted_label"].value_counts()
    total = len(df)
    for label in ["positive", "negative", "neutral"]:
        n = counts.get(label, 0)
        print(f"{label.capitalize():9s}: {n:4d} reviews  ({n/total*100:.1f}%)")

    if "true_label" in df.columns:
        accuracy = (df["predicted_label"] == df["true_label"]).mean()
        print(f"\nAccuracy vs known labels: {accuracy*100:.1f}%")

    print(f"\nAvg sentiment score by product:")
    print(df.groupby("product")["sentiment_score"].mean().sort_values(ascending=False).round(3))
    print("-" * 40)


def chart_sentiment_distribution(df: pd.DataFrame):
    counts = df["predicted_label"].value_counts().reindex(["positive", "neutral", "negative"])
    colors = {"positive": "#16a34a", "neutral": "#6b7280", "negative": "#dc2626"}
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index.str.capitalize(), autopct="%1.0f%%",
            colors=[colors[c] for c in counts.index], startangle=90)
    plt.title("Overall Sentiment Distribution", fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sentiment_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def chart_sentiment_by_product(df: pd.DataFrame):
    pivot = pd.crosstab(df["product"], df["predicted_label"])
    pivot = pivot.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(9, 6),
               color=["#16a34a", "#6b7280", "#dc2626"])
    plt.title("Sentiment Breakdown by Product", fontsize=14, weight="bold")
    plt.xlabel("Product")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sentiment_by_product.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def chart_score_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sentiment_score"], bins=30, kde=True, color="#2563eb")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Distribution of Sentiment Scores (Compound)", fontsize=14, weight="bold")
    plt.xlabel("Sentiment Score (-1 = very negative, +1 = very positive)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "score_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def chart_rating_vs_sentiment(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="rating", y="sentiment_score", hue="rating",
                palette="RdYlGn", legend=False)
    plt.title("Star Rating vs Computed Sentiment Score", fontsize=14, weight="bold")
    plt.xlabel("Star Rating")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "rating_vs_sentiment.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def save_labeled_data(df: pd.DataFrame):
    out = os.path.join(OUTPUT_DIR, "reviews_with_sentiment.csv")
    df.to_csv(out, index=False)
    print(f"Saved labeled dataset: {out}")


if __name__ == "__main__":
    df = load_and_score()
    print_summary(df)
    chart_sentiment_distribution(df)
    chart_sentiment_by_product(df)
    chart_score_distribution(df)
    chart_rating_vs_sentiment(df)
    save_labeled_data(df)
    print("\nAll charts saved in the 'output' folder.")
