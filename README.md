# B2B Laboratory Equipment — Customer Analytics

Customer segmentation and sales-targeting analysis for **PT Gagas Envirotek**. The company had five years of sales history across 300+ accounts and no systematic way to tell which ones to defend, which to grow, and which had already churned quietly.

**Tools:** Python (Pandas, Scikit-Learn, Prophet), SQL, Looker Studio

---

## The problem

The sales team treated all 300+ accounts the same: periodic calls, mass email, equal attention regardless of purchase history. Revenue from a Rp 2.4B account and a one-time Rp 150K purchase looked identical in the spreadsheet. There was no cross-sell intelligence — which customers buy instruments but have never ordered spares; which ones are overdue for a consumable reorder.

## What I built

The project used an 8-stage analysis from raw purchase orders to ranked customer-product opportunities. `scripts/pipeline.py` reconstructs selected core stages for this write-up; the cleaned `notebooks/segmentation.ipynb` preserves project analysis code without its executed outputs. The original client dataset is not included.

```
Stage 1  — Feature extraction
          Customer-level: total revenue, order count, product mix (instruments /
          spares / consumables), recency
          Product-level: buyer count, revenue, average order value

Stage 5  — RFM + 24-month CLTV
          Recency, Frequency, Monetary scores per account
          CLTV model: purchase rate × average order value × estimated lifetime
          Low CLTV helps identify accounts with weaker expected value

Stage 6  — K-Means segmentation
          4 clusters labeled: Champions / Loyal / Growth / Churn
          Separate pass on products to find high-revenue, low-reach items

Stage 7  — Opportunity matrix
          Cross-join: each customer × each product they have never bought
          Score = CLTV × base purchase probability × segment weight × recency decay
          Output: ranked list of (customer, product, opportunity_score)
          Retention targets: accounts with high CLTV but rising recency

Stage 8  — Retention risk scores
          Weighted score combining recency rank, CLTV, and segment label
          Used to prioritise which at-risk accounts need a call this week
```

## Key results

- **300+ accounts** segmented into actionable tiers; ~40% flagged as Churn based on last-order recency and negative CLTV
- **Top cross-sell list** surfaced high-opportunity customer × product pairs the sales team had never systematically identified (e.g., loyal instrument buyers who had never ordered the matching spares)
- **Revenue forecast** for 2026–2027 via Facebook Prophet on monthly revenue series
- **Looker Studio dashboard** delivered to the client as a live report — RFM breakdown, segment distribution, top accounts by CLTV, retention risk leaderboard

## What the output looks like

Each row in the final opportunity list:

```
customer            product                 CLTV_24m    opp_score
Customer A          Example instrument A    40,000,000  5,400,000,000
Customer B          Example instrument A    30,000,000  4,100,000,000
Customer C          Example consumable B    20,000,000  2,800,000,000
```

The sales team gets a short list sorted by opportunity score, not a dump of all 300 accounts. The three rows above are illustrative placeholders, not client output.

## Files in this repo

```
scripts/
  pipeline.py                    # reconstructed core logic: features, RFM/CLTV,
                                  # K-Means segmentation, opportunity scoring
notebooks/
  segmentation.ipynb             # cleaned project notebook; outputs removed
outputs/
  sample_segmentation.csv        # fully synthetic example rows and values
```

The real client dataset and full output (300+ named accounts) are not included. The CSV example uses fabricated values with the same column structure. The notebook preserves original analysis code but has no executed outputs and requires the private input data and path changes before it can run.

## How to run

```bash
pip install pandas scikit-learn
# Place your own order-history export as data/orders.csv with columns:
# customer, sku, product_name, product_type, qty, revenue, order_date
python scripts/pipeline.py
```
