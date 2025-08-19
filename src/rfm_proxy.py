import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from typing import Iterable, Tuple
from src.utils import data_generator, timer_decorator, profile_context


@timer_decorator
def calculate_rfm(
    df: pd.DataFrame,
    customer_id_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate RFM on a single DataFrame (non-streaming)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby(customer_id_col)
        .agg(
            {
                date_col: lambda x: (snapshot_date - x.max()).days,
                customer_id_col: "count",
                amount_col: "sum",
            }
        )
        .rename(
            columns={
                date_col: "Recency",
                customer_id_col: "Frequency",
                amount_col: "Monetary",
            }
        )
        .reset_index()
    )
    return rfm


@timer_decorator
def calculate_rfm_streaming(
    batches: Iterable[pd.DataFrame],
    customer_id_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate RFM from an iterable of DataFrame batches to reduce memory usage."""
    from collections import defaultdict

    last_txn = defaultdict(lambda: pd.NaT)
    freq = defaultdict(int)
    monetary = defaultdict(float)

    for batch in batches:
        b = batch.copy()
        if date_col in b.columns:
            b[date_col] = pd.to_datetime(b[date_col], errors='coerce')
        for cust, grp in b.groupby(customer_id_col):
            max_date = grp[date_col].max()
            if pd.isna(last_txn[cust]) or (max_date is not pd.NaT and max_date > last_txn[cust]):
                last_txn[cust] = max_date
            freq[cust] += len(grp)
            monetary[cust] += grp[amount_col].sum()

    if snapshot_date is None:
        # derive from global max
        snapshot_date = max((d for d in last_txn.values(
        ) if d is not pd.NaT), default=pd.Timestamp.now()) + pd.Timedelta(days=1)

    data = []
    for cust in last_txn.keys():
        recency_days = (
            snapshot_date - last_txn[cust]).days if last_txn[cust] is not pd.NaT else None
        data.append((cust, recency_days, freq[cust], monetary[cust]))

    rfm = pd.DataFrame(
        data, columns=[customer_id_col, 'Recency', 'Frequency', 'Monetary'])
    return rfm


@timer_decorator
def cluster_rfm(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans]:
    """Scale RFM features and cluster customers using KMeans."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]].fillna(0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans


@timer_decorator
def assign_high_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign high risk (1) to the cluster with high recency, low frequency, low monetary value."""
    cluster_stats = rfm.groupby(
        "Cluster")[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats.sort_values(
        ["Frequency", "Monetary", "Recency"], ascending=[True, True, False]
    ).index[0]
    # Functional style mapping
    rfm['is_high_risk'] = list(
        map(lambda c: 1 if c == high_risk_cluster else 0, rfm['Cluster']))
    return rfm[["CustomerId", "is_high_risk"]]


def merge_high_risk(df: pd.DataFrame, high_risk_df: pd.DataFrame, customer_id_col: str = "CustomerId") -> pd.DataFrame:
    """Merge the is_high_risk label back into the main DataFrame."""
    return df.merge(high_risk_df, on=customer_id_col, how="left")


def load_data_auto(input_path: str) -> pd.DataFrame:
    """Try to load as CSV first, then as Excel if CSV fails."""
    try:
        df = pd.read_csv(input_path, encoding="latin1", on_bad_lines="warn")
        print(f"Loaded data as CSV: {input_path}")
        return df
    except Exception as e_csv:
        print(f"CSV load failed: {e_csv}\nTrying Excel...")
        try:
            df = pd.read_excel(input_path)
            print(f"Loaded data as Excel: {input_path}")
            return df
        except Exception as e_xlsx:
            print(f"Excel load failed: {e_xlsx}")
            raise RuntimeError(
                f"Failed to load data as CSV or Excel. Please check your file format.\nCSV error: {e_csv}\nExcel error: {e_xlsx}"
            )


if __name__ == "__main__":
    # Try both .csv and .xlsx
    input_csv = os.path.join("data", "raw", "raw_data.csv")
    input_xlsx = os.path.join("data", "raw", "raw_data.xlsx")
    output_path = os.path.join("data", "processed", "processed_data.csv")

    if os.path.exists(input_csv):
        df = load_data_auto(input_csv)
    elif os.path.exists(input_xlsx):
        df = load_data_auto(input_xlsx)
    else:
        raise FileNotFoundError(
            f"No raw data file found at {input_csv} or {input_xlsx}"
        )

    # Batch RFM calculation using generator for memory efficiency
    with profile_context('rfm_proxy'):
        batches = data_generator(df, batch_size=10000)
        rfm = calculate_rfm_streaming(batches)
        rfm, _ = cluster_rfm(rfm)
        high_risk_df = assign_high_risk(rfm)
        df = merge_high_risk(df, high_risk_df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data with 'is_high_risk' saved to {output_path}")
