import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os


def calculate_rfm(
    df,
    customer_id_col="CustomerId",
    date_col="TransactionStartTime",
    amount_col="Amount",
    snapshot_date=None,
):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for each customer.
    """
    df[date_col] = pd.to_datetime(df[date_col])
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


def cluster_rfm(rfm, n_clusters=3, random_state=42):
    """
    Scale RFM features and cluster customers using KMeans.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans


def assign_high_risk(rfm):
    """
    Assign high risk (1) to the cluster with high recency, low frequency, and low monetary value.
    """
    cluster_stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_stats.sort_values(
        ["Frequency", "Monetary", "Recency"], ascending=[True, True, False]
    ).index[0]
    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)
    return rfm[["CustomerId", "is_high_risk"]]


def merge_high_risk(df, high_risk_df, customer_id_col="CustomerId"):
    """
    Merge the is_high_risk label back into the main DataFrame.
    """
    return df.merge(high_risk_df, on=customer_id_col, how="left")


def load_data_auto(input_path):
    """
    Try to load as CSV first, then as Excel if CSV fails.
    """
    try:
        # Try CSV with latin1 and on_bad_lines for pandas >=1.3
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
    # Prefer CSV, fallback to Excel
    if os.path.exists(input_csv):
        df = load_data_auto(input_csv)
    elif os.path.exists(input_xlsx):
        df = load_data_auto(input_xlsx)
    else:
        raise FileNotFoundError(
            f"No raw data file found at {input_csv} or {input_xlsx}"
        )
    # Calculate RFM
    rfm = calculate_rfm(df)
    # Cluster and assign high risk
    rfm, _ = cluster_rfm(rfm)
    high_risk_df = assign_high_risk(rfm)
    # Merge back
    df = merge_high_risk(df, high_risk_df)
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data with 'is_high_risk' saved to {output_path}")
