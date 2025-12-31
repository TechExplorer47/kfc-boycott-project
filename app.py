import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="KFC Boycott ML Demo", layout="wide")
st.title("🍗 KFC Sales Forecast + Boycott Impact (Demo)")
st.caption("This is simulated/dummy data for a university project (not real KFC financial data).")

# -----------------------------
# Regression part (your 1st cell)
# -----------------------------
def build_regression_data():
    data = {
        "Year": [2018,2019,2020,2021,2022,2023,2024,2025,2026,2027],
        "Sales": [22.0,23.5,24.8,26.3,28.0,23.5,22.8,24.0,26.0,28.5],
        "Boycott": [0,0,0,0,0,2,2,1,1,0],
    }
    return pd.DataFrame(data)

def train_regression(df):
    X = df[["Year", "Boycott"]]
    y = df["Sales"]
    model = LinearRegression().fit(X, y)
    df = df.copy()
    df["Fitted_Sales"] = model.predict(X)
    df["Residuals"] = df["Sales"] - df["Fitted_Sales"]
    return model, df

# -----------------------------
# Simulation part (your big code)
# -----------------------------
def calculate_boycott_impact(date, boycott_start, boycott_peak,
                             boycott_duration, boycott_peak_impact,
                             boycott_long_term_effect, recovery_time):
    boycott_start = pd.to_datetime(boycott_start)
    boycott_peak = pd.to_datetime(boycott_peak)
    current_date = pd.to_datetime(date)

    days_since_start = (current_date - boycott_start).days
    if days_since_start < 0:
        return 1.0

    if days_since_start <= boycott_duration:
        days_to_peak = (boycott_peak - boycott_start).days
        days_to_peak = max(1, days_to_peak)
        if days_since_start <= days_to_peak:
            progress = days_since_start / days_to_peak
            impact = boycott_peak_impact * progress
        else:
            progress = (days_since_start - days_to_peak) / max(1, (boycott_duration - days_to_peak))
            impact = boycott_peak_impact * (1 - progress * 0.5)
        return 1 - impact

    if days_since_start <= boycott_duration + recovery_time:
        progress = (days_since_start - boycott_duration) / max(1, recovery_time)
        strong_impact = boycott_peak_impact * 0.5
        recovery_impact = strong_impact - (strong_impact - boycott_long_term_effect) * progress
        return 1 - recovery_impact

    return 1 - boycott_long_term_effect

def generate_sales_scenario(date_range,
                            base_daily_sales,
                            growth_rate,
                            seasonality_factor,
                            boycott_start,
                            boycott_peak,
                            boycott_duration,
                            boycott_peak_impact,
                            boycott_long_term_effect,
                            recovery_time,
                            with_boycott=True,
                            recovery_level=1.0,
                            seed=42):
    rng = np.random.default_rng(seed)
    daily_sales = []

    for i, date in enumerate(date_range):
        years_from_start = i / 365
        growth_factor = (1 + growth_rate) ** years_from_start
        day_of_year = date.dayofyear
        seasonality = 1 + seasonality_factor * np.sin(2 * np.pi * day_of_year / 365)
        random_factor = 1 + rng.normal(0, 0.03)

        base_sales = base_daily_sales * growth_factor * seasonality * random_factor

        if with_boycott:
            boycott_multiplier = calculate_boycott_impact(
                date, boycott_start, boycott_peak, boycott_duration,
                boycott_peak_impact, boycott_long_term_effect, recovery_time
            )
            boycott_multiplier = 1 - (1 - boycott_multiplier) * recovery_level
        else:
            boycott_multiplier = 1.0

        daily_sales.append(base_sales * boycott_multiplier)

    return np.array(daily_sales)

# -----------------------------
# UI Tabs
# -----------------------------
tab1, tab2 = st.tabs(["1) Regression (ML)", "2) Simulation (Dashboard)"])

with tab1:
    st.subheader("Linear Regression: Sales ~ Year + Boycott")
    df = build_regression_data()
    model, df_fit = train_regression(df)

    st.dataframe(df_fit, use_container_width=True)

    st.markdown("### Future prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        y1 = st.number_input("Year 1", value=2025, step=1)
        b1 = st.number_input("Boycott 1", value=1, step=1)
    with c2:
        y2 = st.number_input("Year 2", value=2026, step=1)
        b2 = st.number_input("Boycott 2", value=1, step=1)
    with c3:
        y3 = st.number_input("Year 3", value=2027, step=1)
        b3 = st.number_input("Boycott 3", value=0, step=1)

    future = pd.DataFrame({"Year": [y1, y2, y3], "Boycott": [b1, b2, b3]})
    future["Predicted_Sales"] = model.predict(future[["Year", "Boycott"]])
    st.dataframe(future, use_container_width=True)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(df_fit["Year"], df_fit["Sales"], label="Actual Sales")
    ax1.plot(df_fit["Year"], df_fit["Fitted_Sales"], linestyle="--", label="Model Trend")
    ax1.plot(future["Year"], future["Predicted_Sales"], linestyle=":", label="Future")
    ax1.set_title("Sales Prediction")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sales")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(df_fit["Year"], df_fit["Residuals"])
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Residuals")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Residuals")

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Daily Sales Simulation (2022–2026)")

    with st.sidebar:
        st.header("Controls")
        base_daily_sales = st.slider("Base daily sales (USD)", 100_000, 5_000_000, 1_000_000, step=50_000)
        growth_rate = st.slider("Annual growth rate", 0.0, 0.20, 0.05, step=0.01)
        seasonality_factor = st.slider("Seasonality amplitude", 0.0, 0.50, 0.15, step=0.01)

        boycott_start = st.date_input("Boycott start", value=pd.to_datetime("2023-10-01")).strftime("%Y-%m-%d")
        boycott_peak = st.date_input("Boycott peak", value=pd.to_datetime("2023-11-15")).strftime("%Y-%m-%d")
        boycott_duration = st.slider("Strong boycott duration (days)", 30, 365, 180, step=10)
        boycott_peak_impact = st.slider("Peak impact (drop %)", 0.0, 0.95, 0.65, step=0.05)
        boycott_long_term_effect = st.slider("Long-term effect (drop %)", 0.0, 0.50, 0.15, step=0.01)
        recovery_time = st.slider("Recovery time (days)", 30, 730, 365, step=10)
        continued_recovery_level = st.slider("Continued boycott severity (0..1)", 0.1, 1.0, 0.7, step=0.05)
        seed = st.number_input("Random seed", value=42, step=1)

    date_range = pd.date_range(start="2022-01-01", end="2026-12-31", freq="D")

    sales_with = generate_sales_scenario(
        date_range, base_daily_sales, growth_rate, seasonality_factor,
        boycott_start, boycott_peak, boycott_duration,
        boycott_peak_impact, boycott_long_term_effect, recovery_time,
        with_boycott=True, recovery_level=1.0, seed=seed
    )
    sales_no = generate_sales_scenario(
        date_range, base_daily_sales, growth_rate, seasonality_factor,
        boycott_start, boycott_peak, boycott_duration,
        boycott_peak_impact, boycott_long_term_effect, recovery_time,
        with_boycott=False, recovery_level=1.0, seed=seed
    )
    sales_cont = generate_sales_scenario(
        date_range, base_daily_sales, growth_rate, seasonality_factor,
        boycott_start, boycott_peak, boycott_duration,
        boycott_peak_impact, boycott_long_term_effect, recovery_time,
        with_boycott=True, recovery_level=continued_recovery_level, seed=seed
    )

    sim_df = pd.DataFrame({
        "Date": date_range,
        "With_Boycott": sales_with,
        "No_Boycott": sales_no,
        "Continued_Boycott": sales_cont,
    })

    fig2 = plt.figure(figsize=(14, 6))
    plt.plot(sim_df["Date"], sim_df["No_Boycott"]/1e6, label="No Boycott", linewidth=2)
    plt.plot(sim_df["Date"], sim_df["With_Boycott"]/1e6, label="Actual Boycott", linewidth=2)
    plt.plot(sim_df["Date"], sim_df["Continued_Boycott"]/1e6, label="Continued Boycott", linestyle="--", linewidth=2)
    plt.axvline(pd.to_datetime(boycott_start), linestyle=":", label="Start")
    plt.axvline(pd.to_datetime(boycott_peak), linestyle=":", label="Peak")
    plt.fill_between(sim_df["Date"], sim_df["No_Boycott"]/1e6, sim_df["With_Boycott"]/1e6, alpha=0.12, label="Loss")
    plt.title("Boycott Impact Simulation")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales (Million $)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)

    total_no = sim_df["No_Boycott"].sum()
    total_with = sim_df["With_Boycott"].sum()
    total_cont = sim_df["Continued_Boycott"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total (No Boycott)", f"${total_no/1e9:.2f}B")
    c2.metric("Total (Actual Boycott)", f"${total_with/1e9:.2f}B")
    c3.metric("Total (Continued Boycott)", f"${total_cont/1e9:.2f}B")

    st.download_button(
        "Download simulated daily data (CSV)",
        data=sim_df.to_csv(index=False).encode("utf-8"),
        file_name="kfc_boycott_simulation.csv",
        mime="text/csv",
    )
