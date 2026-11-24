import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Benford's Law Forensic Dashboard", layout="wide")
st.title("Benford's Law Statistical Analysis Dashboard")
st.markdown(
    "Upload a CSV or Excel dataset, apply dynamic filters, and test numeric columns "
    "against Benford's Law (1st Digit and First-2 Digits) using the NumPy Kolmogorov-Smirnov (KS) test."
)

# ---------------------------------------------------------
# 1. File Upload
# ---------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "1. Upload CSV or Excel File", type=["csv", "xlsx", "xls"]
)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

df = load_data(uploaded_file)

# ---------------------------------------------------------
# 2. Dynamic Filtering - At Least Three Criteria
# ---------------------------------------------------------
st.sidebar.header("2. Filter Data (3+ Criteria)")
all_cols = df.columns.tolist()

default_filter_cols = all_cols[:3] if len(all_cols) >= 3 else all_cols
selected_filter_cols = st.sidebar.multiselect(
    "Select at least 3 columns to filter by:",
    options=all_cols,
    default=default_filter_cols,
)

if len(selected_filter_cols) < 3:
    st.warning("Please select at least 3 filter criteria in the sidebar to satisfy dashboard requirements.")

filtered_df = df.copy()

for col in selected_filter_cols:
    if pd.api.types.is_numeric_dtype(filtered_df[col]):
        clean_series = filtered_df[col].dropna()
        if not clean_series.empty:
            min_val, max_val = float(clean_series.min()), float(clean_series.max())
            if min_val < max_val:
                step = (max_val - min_val) / 100.0
                chosen_range = st.sidebar.slider(
                    f"Filter {col} (Range)",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                )
                filtered_df = filtered_df[
                    filtered_df[col].between(chosen_range[0], chosen_range[1])
                ]
    else:
        unique_vals = filtered_df[col].dropna().astype(str).unique().tolist()
        chosen_vals = st.sidebar.multiselect(
            f"Filter {col} (Categories)",
            options=sorted(unique_vals),
            default=sorted(unique_vals),
        )
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(chosen_vals)]

st.sidebar.markdown(f"**Filtered Rows:** {len(filtered_df):,} / {len(df):,}")

# ---------------------------------------------------------
# 3. Target Column Selection & Benford / NumPy KS Logic
# ---------------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns available in the uploaded dataset for Benford analysis.")
    st.stop()

target_col = st.selectbox("Select Numeric Column of Interest for Benford Analysis:", numeric_cols)

def extract_leading_digits(series: pd.Series, num_digits: int = 1) -> np.ndarray:
    """Extracts 1st digit (1-9) or first-2 digits (10-99) from non-zero values."""
    vals = np.abs(series.dropna().to_numpy(dtype=float))
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.array([], dtype=int)
    
    exponents = np.floor(np.log10(vals))
    mantissas = vals / (10.0 ** exponents)
    
    if num_digits == 1:
        digits = np.floor(mantissas).astype(int)
        return digits[(digits >= 1) & (digits <= 9)]
    else:
        digits = np.floor(mantissas * 10.0).astype(int)
        return digits[(digits >= 10) & (digits <= 99)]

def run_benford_ks_numpy(digits: np.ndarray, num_digits: int = 1):
    """
    Computes Observed vs Expected Benford frequencies and performs the
    Kolmogorov-Smirnov (KS) test strictly using NumPy.
    """
    if num_digits == 1:
        digit_range = np.arange(1, 10)
    else:
        digit_range = np.arange(10, 100)

    n = len(digits)
    benford_probs = np.log10(1.0 + (1.0 / digit_range))

    counts = np.array([np.sum(digits == d) for d in digit_range], dtype=float)
    actual_probs = counts / n if n > 0 else np.zeros_like(benford_probs)

    actual_cdf = np.cumsum(actual_probs)
    benford_cdf = np.cumsum(benford_probs)

    abs_diff_cdf = np.abs(actual_cdf - benford_cdf)
    ks_stat = float(np.max(abs_diff_cdf))

    crit_05 = 1.36 / np.sqrt(n) if n > 0 else np.nan
    crit_01 = 1.63 / np.sqrt(n) if n > 0 else np.nan
    is_different_05 = bool(ks_stat > crit_05)

    summary_table = pd.DataFrame({
        "Leading_Digits": digit_range,
        "Observed_Count": counts.astype(int),
        "Actual_Pct": np.round(actual_probs * 100, 3),
        "Benford_Pct": np.round(benford_probs * 100, 3),
        "Diff_Pct": np.round((actual_probs - benford_probs) * 100, 3),
        "Actual_CDF": np.round(actual_cdf, 4),
        "Benford_CDF": np.round(benford_cdf, 4),
        "Abs_CDF_Diff": np.round(abs_diff_cdf, 4),
    })

    stats_dict = {
        "n": n,
        "ks_stat": ks_stat,
        "crit_05": crit_05,
        "crit_01": crit_01,
        "statistically_different": is_different_05,
    }
    return summary_table, stats_dict

digits_1 = extract_leading_digits(filtered_df[target_col], num_digits=1)
digits_2 = extract_leading_digits(filtered_df[target_col], num_digits=2)

if len(digits_1) == 0:
    st.warning("Selected column has no positive non-zero values after filtering.")
    st.stop()

df_1st, stats_1st = run_benford_ks_numpy(digits_1, num_digits=1)
df_2nd, stats_2nd = run_benford_ks_numpy(digits_2, num_digits=2)

# ---------------------------------------------------------
# 4. Show Statistical Analysis Results
# ---------------------------------------------------------
st.header("Statistical Test Results (NumPy KS Statistic)")
col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.subheader("First-Digit Analysis (1–9)")
    st.metric("Sample Size (N)", f"{stats_1st['n']:,}")
    st.metric("KS Statistic (D)", f"{stats_1st['ks_stat']:.4f}")
    st.write(f"**Critical Value (α = 0.05):** `{stats_1st['crit_05']:.4f}`")
    if stats_1st["statistically_different"]:
        st.error(
            f"**Conclusion:** Statistically DIFFERENT from Benford's Law "
            f"(KS {stats_1st['ks_stat']:.4f} > {stats_1st['crit_05']:.4f})."
        )
    else:
        st.success(
            f"**Conclusion:** Conforms to Benford's Law — NOT statistically different "
            f"(KS {stats_1st['ks_stat']:.4f} ≤ {stats_1st['crit_05']:.4f})."
        )

with col_stat2:
    st.subheader("First-Two Digits Analysis (10–99)")
    st.metric("Sample Size (N)", f"{stats_2nd['n']:,}")
    st.metric("KS Statistic (D)", f"{stats_2nd['ks_stat']:.4f}")
    st.write(f"**Critical Value (α = 0.05):** `{stats_2nd['crit_05']:.4f}`")
    if stats_2nd["statistically_different"]:
        st.error(
            f"**Conclusion:** Statistically DIFFERENT from Benford's Law "
            f"(KS {stats_2nd['ks_stat']:.4f} > {stats_2nd['crit_05']:.4f})."
        )
    else:
        st.success(
            f"**Conclusion:** Conforms to Benford's Law — NOT statistically different "
            f"(KS {stats_2nd['ks_stat']:.4f} ≤ {stats_2nd['crit_05']:.4f})."
        )

# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------
st.header("Visual Comparison: Actual vs. Benford Distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

x1 = df_1st["Leading_Digits"]
ax1.bar(x1 - 0.18, df_1st["Actual_Pct"], width=0.36, label="Actual %", color="#2b5c8f")
ax1.bar(x1 + 0.18, df_1st["Benford_Pct"], width=0.36, label="Benford Expected %", color="#e07a5f")
ax1.set_title(f"1st Digit Distribution ({target_col})\nKS = {stats_1st['ks_stat']:.4f} (Crit 0.05 = {stats_1st['crit_05']:.4f})")
ax1.set_xlabel("First Digit (1–9)")
ax1.set_ylabel("Frequency (%)")
ax1.set_xticks(x1)
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.4)

x2 = df_2nd["Leading_Digits"]
ax2.bar(x2, df_2nd["Actual_Pct"], width=0.7, label="Actual %", color="#2b5c8f", alpha=0.85)
ax2.plot(x2, df_2nd["Benford_Pct"], label="Benford Expected %", color="#d62828", linewidth=2)
ax2.set_title(f"First-2 Digits Distribution ({target_col})\nKS = {stats_2nd['ks_stat']:.4f} (Crit 0.05 = {stats_2nd['crit_05']:.4f})")
ax2.set_xlabel("First Two Digits (10–99)")
ax2.set_ylabel("Frequency (%)")
ax2.set_xticks(np.arange(10, 100, 10))
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
st.pyplot(fig)

tab1, tab2, tab3 = st.tabs([
    "1st Digit Summary Table",
    "First-2 Digits Summary Table",
    "Filtered Dataset Preview"
])
with tab1:
    st.dataframe(df_1st, use_container_width=True)
with tab2:
    st.dataframe(df_2nd, use_container_width=True)
with tab3:
    st.dataframe(filtered_df.head(200), use_container_width=True)

# ---------------------------------------------------------
# 6. Download and Share Results
# ---------------------------------------------------------
st.header("Download & Share Analysis")

img_buffer = io.BytesIO()
fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
img_buffer.seek(0)

report_text = f"""# Benford's Law Statistical Analysis Report
- **Source File:** {uploaded_file.name}
- **Analyzed Column:** {target_col}
- **Filtered Sample Size (N):** {stats_1st['n']:,} (out of {len(df):,} total rows)
- **Active Filter Columns:** {', '.join(selected_filter_cols)}

## 1st Digit KS Test Results
- **KS Statistic (D):** {stats_1st['ks_stat']:.4f}
- **Critical Value (α = 0.05):** {stats_1st['crit_05']:.4f}
- **Statistically Different from Benford?** {'Yes (Reject Null)' if stats_1st['statistically_different'] else 'No (Conforms to Benford)'}

## First-2 Digits KS Test Results
- **KS Statistic (D):** {stats_2nd['ks_stat']:.4f}
- **Critical Value (α = 0.05):** {stats_2nd['crit_05']:.4f}
- **Statistically Different from Benford?** {'Yes (Reject Null)' if stats_2nd['statistically_different'] else 'No (Conforms to Benford)'}
"""

dl1, dl2, dl3, dl4 = st.columns(4)

with dl1:
    st.download_button(
        label="Download 1st-Digit Summary (CSV)",
        data=df_1st.to_csv(index=False).encode("utf-8"),
        file_name=f"benford_1st_digit_{target_col}.csv",
        mime="text/csv",
    )

with dl2:
    st.download_button(
        label="Download 2-Digit Summary (CSV)",
        data=df_2nd.to_csv(index=False).encode("utf-8"),
        file_name=f"benford_2_digits_{target_col}.csv",
        mime="text/csv",
    )

with dl3:
    st.download_button(
        label="Download Charts (PNG)",
        data=img_buffer,
        file_name=f"benford_charts_{target_col}.png",
        mime="image/png",
    )

with dl4:
    st.download_button(
        label="Download Summary Report (MD)",
        data=report_text.encode("utf-8"),
        file_name=f"benford_report_{target_col}.md",
        mime="text/markdown",
    )