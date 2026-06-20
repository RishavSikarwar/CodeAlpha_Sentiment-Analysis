"""
Generates a realistic sample dataset of product reviews for sentiment
analysis practice. In a real-world project, this CSV would instead come
from scraping Amazon reviews, pulling tweets/social media posts via an
API, or exporting customer feedback from a survey tool.

Run:
    python generate_dataset.py
"""

import os
import csv
import random
from datetime import date, timedelta

random.seed(7)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "product_reviews.csv")

PRODUCTS = ["Wireless Earbuds", "Smartwatch", "Air Fryer", "Running Shoes", "Backpack"]

POSITIVE_TEMPLATES = [
    "Absolutely love this {p}! Works perfectly and exceeded my expectations.",
    "Great {p}, very happy with the quality and fast delivery.",
    "This {p} is amazing, best purchase I've made this year.",
    "Excellent build quality and the {p} performs really well.",
    "I'm impressed with this {p}, highly recommend it to everyone.",
    "Super comfortable and reliable {p}, worth every penny.",
    "Fantastic {p}! Battery life and performance are outstanding.",
]
NEGATIVE_TEMPLATES = [
    "Terrible {p}, stopped working after two days. Very disappointed.",
    "Waste of money, this {p} broke within a week.",
    "Poor quality {p}, not what I expected at all.",
    "I hate this {p}, customer service was no help either.",
    "The {p} arrived damaged and the return process was a nightmare.",
    "Worst {p} I've ever bought, completely useless.",
    "Disappointed with this {p}, it feels cheap and flimsy.",
]
NEUTRAL_TEMPLATES = [
    "The {p} is okay, does what it says but nothing special.",
    "Received the {p} on time, packaging was standard.",
    "This {p} is average, similar to other ones I've used.",
    "It's a {p}, works as described, no major complaints.",
    "The {p} matches the description, decent for the price.",
    "Functional {p}, but the design could be improved.",
]

START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)


def random_date():
    delta = (END_DATE - START_DATE).days
    return START_DATE + timedelta(days=random.randint(0, delta))


def generate_rows(n=300):
    rows = []
    for i in range(1, n + 1):
        product = random.choice(PRODUCTS)
        bucket = random.choices(
            ["positive", "negative", "neutral"], weights=[0.5, 0.3, 0.2]
        )[0]
        template = random.choice(
            POSITIVE_TEMPLATES if bucket == "positive"
            else NEGATIVE_TEMPLATES if bucket == "negative"
            else NEUTRAL_TEMPLATES
        )
        review_text = template.format(p=product)
        rating = (
            random.choice([4, 5]) if bucket == "positive"
            else random.choice([1, 2]) if bucket == "negative"
            else 3
        )
        rows.append({
            "review_id": f"REV{i:04d}",
            "product": product,
            "review_date": random_date().isoformat(),
            "rating": rating,
            "review_text": review_text,
            # true_label kept only to validate our sentiment model later
            "true_label": bucket,
        })
    return rows


if __name__ == "__main__":
    rows = generate_rows()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} reviews -> {OUTPUT_FILE}")
