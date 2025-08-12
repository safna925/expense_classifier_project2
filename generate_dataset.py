# generate_dataset.py
import pandas as pd
import random

# Categories
categories = [
    "Charity", "Education", "Entertainment", "Food", "Groceries",
    "Healthcare", "Insurance", "Rent", "Shopping", "Transportation",
    "Travel", "Utilities"
]

# Category-specific, context-rich keywords (no dangerous overlaps)
keywords = {
    "Charity": [
        "donation to ngo", "charity fundraiser event", "orphanage relief fund",
        "ngo food supply", "flood relief charity drive", "animal shelter charity",
        "environmental fundraiser", "charity for children"
    ],
    "Education": [
        "college semester fee", "school annual tuition", "online coding course",
        "university admission payment", "exam registration form fee",
        "library membership renewal", "engineering workshop fee",
        "language learning subscription"
    ],
    "Entertainment": [
        "bookmyshow movie ticket", "live concert pass", "netflix annual subscription",
        "spotify music plan", "standup comedy ticket", "amusement park entry pass",
        "theatre drama booking", "video game purchase steam"
    ],
    "Food": [
        "zomato chicken biryani", "restaurant dosa breakfast", "burger king combo meal",
        "swiggy pizza delivery", "dominos cheese burst order",
        "street food chaat payment", "cafe cappuccino order", "kfc crispy bucket meal"
    ],
    "Groceries": [
        "buy fresh vegetables", "bigbasket organic fruits", "milk bread eggs purchase",
        "dmart monthly grocery shopping", "supermarket rice and dal",
        "local market spinach and tomato", "more supermarket butter and cheese",
        "organic store flour and oil"
    ],
    "Healthcare": [
        "hospital surgery bill", "doctor specialist consultation",
        "pharmacy antibiotic medicine", "eye hospital laser treatment",
        "dental root canal payment", "medical lab blood test",
        "orthopedic checkup visit", "covid vaccination center payment"
    ],
    "Insurance": [
        "life insurance yearly premium", "car insurance renewal fee",
        "health insurance claim settlement", "travel insurance package purchase",
        "home insurance policy coverage", "bike insurance payment online",
        "term insurance policy fee", "medical insurance plan renewal"
    ],
    "Rent": [
        "monthly apartment rent", "pg hostel room fee", "flat lease agreement payment",
        "landlord rent deposit", "office space monthly rent", "shop rental payment",
        "house rent cash payment", "villa rent for holiday month"
    ],
    "Shopping": [
        "amazon electronics purchase", "flipkart clothing order",
        "zara summer dress shopping", "nike sports shoes order",
        "shopping mall handbag purchase", "h&m t-shirt pack order",
        "adidas track pants buy", "apple store macbook purchase"
    ],
    "Transportation": [
        "uber ride fare", "ola cab booking", "metro train ticket",
        "petrol station fuel refill", "diesel car tank refill",
        "bus pass monthly recharge", "bike taxi ride payment", "airport parking fee"
    ],
    "Travel": [
        "international flight booking", "hotel resort stay",
        "holiday tour package booking", "beach resort vacation stay",
        "mountain trekking trip plan", "train reservation sleeper class",
        "cruise ship vacation booking", "city travel sightseeing tour"
    ],
    "Utilities": [
        "electricity bill payment online", "wifi broadband recharge",
        "water pipeline usage bill", "cooking gas connection refill",
        "mobile postpaid bill payment", "sewage service fee",
        "landline telephone bill", "solar power maintenance payment"
    ]
}

# Small percentage of ambiguous phrases with clear context words
ambiguous_phrases = [
    ("school trip bus ticket", ["Education", "Transportation"]),
    ("charity food distribution event", ["Charity", "Food"]),
    ("health insurance premium medical", ["Insurance", "Healthcare"]),
    ("hotel dinner buffet", ["Travel", "Food"]),
    ("movie night snacks order", ["Entertainment", "Food"]),
    ("shopping mall grocery store", ["Shopping", "Groceries"])
]

entries = []
random.seed(42)
total_entries = 10000
entries_per_category = total_entries // len(categories)

for cat in categories:
    count = 0
    while count < entries_per_category:
        if random.random() < 0.05:  # ~5% ambiguous data
            phrase, possible_cats = random.choice(ambiguous_phrases)
            assigned_cat = random.choice(possible_cats)
        else:
            phrase = random.choice(keywords[cat])
            assigned_cat = cat

        # Add small variations for robustness
        if random.random() < 0.15:
            phrase += " " + random.choice(["offer", "deal", "payment", "online", "order", "service", "booking"])

        entries.append([phrase, assigned_cat])
        count += 1

# Shuffle dataset
random.shuffle(entries)

# Save dataset
df = pd.DataFrame(entries, columns=["text", "category"])
df.to_csv("transactions.csv", index=False)

print(f"✅ Dataset generated: {len(df)} entries, {len(categories)} categories, {entries_per_category} per category")
