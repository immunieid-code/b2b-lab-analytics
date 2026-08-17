# -*- coding: utf-8 -*-
"""
Customer segmentation & opportunity-scoring pipeline for a B2B lab equipment
distributor. Reconstructed for this write-up to match the real column names
and stage logic used on the original project; the original client dataset
is not included here (confidentiality) — run this against your own
order-history export with the same shape (see README for the expected columns).
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Stage 1 — customer & product features from raw purchase orders
# ---------------------------------------------------------------------------
def build_customer_features(orders: pd.DataFrame) -> pd.DataFrame:
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    today = orders["order_date"].max()

    features = orders.groupby("customer").agg(
        total_revenue=("revenue", "sum"),
        total_qty=("qty", "sum"),
        avg_order_value=("revenue", "mean"),
        last_order_date=("order_date", "max"),
    )
    features["recency_days"] = (today - features["last_order_date"]).dt.days

    # product-mix columns: how many distinct instrument / sparepart /
    # consumable SKUs each customer has ever bought
    mix = orders.pivot_table(
        index="customer", columns="product_type", values="sku",
        aggfunc="nunique", fill_value=0,
    )
    return features.join(mix).reset_index()


# ---------------------------------------------------------------------------
# Stage 5 — RFM scores + 24-month CLTV
# ---------------------------------------------------------------------------
def add_rfm_cltv(features: pd.DataFrame, months: int = 24) -> pd.DataFrame:
    df = features.copy()
    purchase_rate = df["total_qty"] / df["recency_days"].clip(lower=1) * 30
    df[f"CLTV_{months}m"] = purchase_rate * df["avg_order_value"] * months
    return df


# ---------------------------------------------------------------------------
# Stage 6 — K-Means segmentation (Champions / Loyal / Growth / Risk / Churn)
# ---------------------------------------------------------------------------
def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    cols = ["total_revenue", "recency_days", "total_qty"]
    scaled = StandardScaler().fit_transform(df[cols])
    df["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    # label clusters by their recency/revenue centroid rather than hard-coding
    # cluster index -> name, since KMeans cluster numbering isn't stable
    centroids = df.groupby("cluster")[cols].mean()
    order = centroids.sort_values("recency_days").index.tolist()
    labels = {order[0]: "Champions", order[1]: "Loyal", order[2]: "Growth", order[3]: "Churn"}
    df["segment"] = df["cluster"].map(labels)
    return df


# ---------------------------------------------------------------------------
# Stage 7 — opportunity matrix: score every (customer, product) pair the
# customer has never bought, so sales knows exactly what to pitch next.
# ---------------------------------------------------------------------------
SEGMENT_WEIGHT = {"Champions": 1.5, "Loyal": 1.3, "Growth": 1.1, "Risk": 0.7, "Churn": 0.3}

def build_opportunity_matrix(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    already_bought = set(zip(orders["customer"], orders["sku"]))
    rows = []
    for _, cust in customers.iterrows():
        for _, prod in products.iterrows():
            if (cust["customer"], prod["sku"]) in already_bought:
                continue
            recency_decay = 1 / (1 + cust["recency_days"] / 180)
            base_probability = 0.5
            score = (
                cust["CLTV_24m"]
                * base_probability
                * SEGMENT_WEIGHT.get(cust["segment"], 1.0)
                * recency_decay
            )
            rows.append({
                "customer": cust["customer"],
                "product": prod["product_name"],
                "opportunity_score": score,
            })
    return pd.DataFrame(rows).sort_values("opportunity_score", ascending=False)


if __name__ == "__main__":
    orders = pd.read_csv("data/orders.csv")
    products = orders[["sku", "product_name"]].drop_duplicates()

    features = build_customer_features(orders)
    features = add_rfm_cltv(features)
    features = segment_customers(features)

    opportunities = build_opportunity_matrix(features, products, orders)

    features.to_csv("outputs/customer_segmentation.csv", index=False)
    opportunities.head(200).to_csv("outputs/top_recommendations.csv", index=False)
    print(f"Segmented {len(features)} customers, scored {len(opportunities)} opportunities.")
