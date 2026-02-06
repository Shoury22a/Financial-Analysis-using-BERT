import pandas as pd
import os

# COMPREHENSIVE FINANCIAL SENTIMENT DATASET
# Goal: Train model on IMPLICIT meaning, not just keywords

data = [
    # ===================== POSITIVE (0) =====================
    # --- User's Original Explicit Positive ---
    ("The company reported a strong quarterly profit exceeding analyst expectations.", 0),
    ("Revenue grew by 25% year-over-year, driven by higher product demand.", 0),
    ("The stock price surged after the successful product launch.", 0),
    ("Management announced a strategic partnership expected to boost long-term growth.", 0),
    ("The firm achieved record-breaking earnings this fiscal year.", 0),
    ("Investors showed strong confidence, pushing the share value higher.", 0),
    ("Cost-cutting initiatives significantly improved operating margins.", 0),
    ("The company's outlook remains optimistic with expanding global markets.", 0),
    ("Dividend payments were increased for the third consecutive year.", 0),
    ("Analysts upgraded the stock rating to buy due to strong fundamentals.", 0),
    
    # --- User's NEW Implicit Positive (CRITICAL for generalization) ---
    ("The company exceeded analyst revenue estimates for the third consecutive quarter.", 0),
    ("Cash reserves increased while operating expenses remained stable.", 0),
    ("Demand for the new product continued to outpace production capacity.", 0),
    ("The firm expanded into three additional international markets this year.", 0),
    ("Shareholders approved an increase in dividend distribution.", 0),
    ("Customer subscriptions grew steadily throughout the fiscal period.", 0),
    ("Debt obligations were reduced ahead of schedule.", 0),
    ("The latest earnings report triggered heavy institutional buying.", 0),
    ("Production efficiency improvements lowered overall unit costs.", 0),
    ("The company secured multiple long-term government contracts.", 0),
    
    # --- Additional Implicit Positive Variations ---
    ("Apple has skyrocketed growth.", 0),  # User's specific failing case
    ("Tesla reported better than expected deliveries.", 0),
    ("The acquisition will be accretive to earnings.", 0),
    ("Market share gains accelerated during the quarter.", 0),
    ("The company raised its full-year guidance.", 0),
    ("Profit margins expanded despite inflationary pressures.", 0),
    ("Free cash flow generation exceeded all expectations.", 0),
    ("The backlog of orders reached an all-time high.", 0),
    ("Return on equity improved significantly.", 0),
    ("Customer retention rates hit record levels.", 0),
    ("The upgrade cycle is driving strong demand.", 0),
    ("Operating leverage is improving.", 0),
    ("The company is gaining pricing power.", 0),
    ("Unit economics are trending favorably.", 0),
    ("The turnaround strategy is showing results.", 0),

    # ===================== NEGATIVE (1) =====================
    # --- User's Original Explicit Negative ---
    ("The company posted a significant quarterly loss due to declining sales.", 1),
    ("Revenue fell short of market expectations, causing the stock to drop.", 1),
    ("Rising debt levels are creating financial instability.", 1),
    ("The firm announced mass layoffs to reduce operational costs.", 1),
    ("Profit margins shrunk because of increasing raw material prices.", 1),
    ("Investors reacted negatively to the weak earnings guidance.", 1),
    ("The stock price plunged after regulatory concerns emerged.", 1),
    ("Cash flow problems indicate serious liquidity risks.", 1),
    ("The company is facing lawsuits that may hurt future earnings.", 1),
    ("Credit rating agencies downgraded the firm's outlook to negative.", 1),
    
    # --- User's NEW Implicit Negative (CRITICAL for generalization) ---
    ("Operating expenses rose faster than total revenue.", 1),
    ("The firm delayed its expansion plans due to funding constraints.", 1),
    ("Several senior executives resigned within the same quarter.", 1),
    ("Inventory levels continued to accumulate in warehouses.", 1),
    ("The company required additional borrowing to meet short-term obligations.", 1),
    ("Regulatory authorities initiated an investigation into accounting practices.", 1),
    ("Customer cancellations increased compared to the previous quarter.", 1),
    ("Production facilities remained underutilized for most of the year.", 1),
    ("Suppliers shortened payment timelines citing credit concerns.", 1),
    ("The organization suspended dividend distribution until further notice.", 1),
    
    # --- Additional Implicit Negative Variations ---
    ("Growth is slowing quarter over quarter.", 1),
    ("The company missed consensus estimates.", 1),
    ("Management lowered its full-year outlook.", 1),
    ("Churn rates are increasing.", 1),
    ("The balance sheet is becoming stretched.", 1),
    ("Working capital requirements are rising.", 1),
    ("The company is burning cash.", 1),
    ("Competitive pressures are intensifying.", 1),
    ("Market share losses are accelerating.", 1),
    ("The company faces headwinds from currency fluctuations.", 1),
    ("Auditors raised going concern warnings.", 1),
    ("The CFO departure signals internal issues.", 1),
    ("Order cancellations are mounting.", 1),
    ("The product recall will impact earnings.", 1),
    ("Restructuring charges will weigh on results.", 1),

    # ===================== NEUTRAL (2) =====================
    # --- User's Original Neutral ---
    ("The company released its quarterly financial results today.", 2),
    ("Revenue remained unchanged compared to last year.", 2),
    ("Management announced a new board member appointment.", 2),
    ("The firm plans to expand operations into Asia next year.", 2),
    ("Shares closed flat in today's trading session.", 2),
    ("The annual shareholder meeting will be held next month.", 2),
    ("The company disclosed its capital expenditure plans.", 2),
    ("A merger discussion is currently under review.", 2),
    ("The organization published its sustainability report.", 2),
    ("Analysts are monitoring the upcoming earnings call.", 2),
    
    # --- User's NEW Truly Neutral (CRITICAL for generalization) ---
    ("The board meeting is scheduled for next Monday.", 2),
    ("The company operates in the renewable energy sector.", 2),
    ("Annual revenue figures were published in the official report.", 2),
    ("The firm maintains offices in five different countries.", 2),
    ("Shares were traded across major global exchanges today.", 2),
    ("The earnings call lasted approximately one hour.", 2),
    ("A new compliance policy was introduced this quarter.", 2),
    ("Management presented the long-term strategic roadmap.", 2),
    ("The organization employs over 10,000 people worldwide.", 2),
    ("Financial statements were filed with the regulatory authority.", 2),
    
    # --- Additional Neutral Variations ---
    ("The company is headquartered in California.", 2),
    ("The CEO will present at the investor conference.", 2),
    ("Trading volume was in line with the 30-day average.", 2),
    ("The company has been public since 1998.", 2),
    ("A new product line will be announced next quarter.", 2),
    ("The fiscal year ends in December.", 2),
    ("The company is part of the S&P 500 index.", 2),
    ("Results will be reported after market close.", 2),
    ("The analyst day is scheduled for November.", 2),
    ("The company operates a subscription-based model.", 2),
    ("A regular quarterly dividend was declared.", 2),
    ("Stock options were granted to employees.", 2),
    ("The company completed a refinancing transaction.", 2),
    ("A new warehouse facility was opened.", 2),
    ("The patent application is pending review.", 2),

    # ===================== MIXED/HARD (Map to dominant implication) =====================
    # --- User's Original Mixed ---
    ("Despite higher revenue, net profit declined due to increased expenses.", 1),
    ("The company showed moderate growth but warned about future uncertainty.", 1),
    ("Strong demand boosted sales, yet supply chain disruptions limited profits.", 1),
    ("Earnings beat expectations, although management provided cautious guidance.", 1),
    ("Revenue improved slightly, but debt levels remain concerning.", 1),
    
    # --- User's NEW Hard/Mixed ---
    ("Revenue increased, while capital expenditure also rose significantly.", 2),  # Neutral - offsetting factors
    ("The company entered new markets but required substantial upfront investment.", 2),  # Neutral - trade-off
    ("Subscriber growth continued alongside higher service maintenance costs.", 2),  # Neutral - mixed
    ("Borrowing levels expanded to support ongoing infrastructure projects.", 2),  # Neutral - context-dependent
    ("Sales volumes improved even as pricing pressure persisted.", 0),  # Slightly Positive - sales UP is key
    
    # --- More Hard Cases ---
    ("While costs increased, net profit soared by 20%.", 0),
    ("Despite the recall, sales remained robust.", 0),
    ("Revenue was flat but margins expanded.", 0),
    ("Losses narrowed compared to the prior quarter.", 0),  # Improving bad situation = slightly positive
    ("The company is profitable but growth is slowing.", 1),  # Slowing growth = slightly negative for high-growth
]


# --- AUGMENTATION WITH USER CSV ---
def content_filter(text):
    text = str(text).lower()
    
    # Strict Keywords for Auto-Labeling (High Precision)
    pos_keywords = [
        "surge", "soar", "jump", "record high", "beat estimates", "growth", 
        "profit rise", "gain", "bullish", "upgrade", "optimistic", "strong",
        "outperform", "buy rating", "rally", "upbeat", "higher", "positive"
    ]
    
    neg_keywords = [
        "plunge", "dive", "drop", "miss", "loss", "crash", "bearish", 
        "downgrade", "weak", "concern", "risk", "uncertainty", "down", 
        "fall", "lower", "negative", "pressure", "decline", "tumble", "sell-off"
    ]

    for k in pos_keywords:
        if k in text: return 0 # Positive
    
    for k in neg_keywords:
        if k in text: return 1 # Negative
        
    return None # Skip ambiguous

try:
    csv_path = "scraped_business_insider_news.csv"
    if os.path.exists(csv_path):
        print(f"Loading extra data from {csv_path}...")
        user_df = pd.read_csv(csv_path)
        
        extra_samples = []
        for index, row in user_df.iterrows():
            # Check headline
            label = content_filter(row.get('headline', ''))
            if label is not None:
                extra_samples.append((row['headline'], label))
            
            # Check content (first 200 chars to avoid noise)
            content_snippet = str(row.get('content', ''))[:200]
            label_c = content_filter(content_snippet)
            if label_c is not None:
                 extra_samples.append((content_snippet, label_c))

        print(f"-> Extracted {len(extra_samples)} high-confidence samples from CSV.")
        data.extend(extra_samples)
    else:
        print("CSV file not found, skipping augmentation.")
except Exception as e:
    print(f"Error reading CSV: {e}")

# Create final dataframe
df = pd.DataFrame(data, columns=["sentence", "label"])

# Shuffle to prevent ordering bias during training
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("financial_news_augmented.csv", index=False)
print(f"FINAL DATASET SIZE: {len(df)} samples.")
print(f"Positive: {(df['label']==0).sum()}, Negative: {(df['label']==1).sum()}, Neutral: {(df['label']==2).sum()}")
