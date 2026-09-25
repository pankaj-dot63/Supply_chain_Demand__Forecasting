import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

MODEL_PATH = Path("demand_forecasting_model.pkl")
# NOTE: filename matches the notebook's saved artifact exactly ("eatures.pkl"),
# do not "fix" the spelling or the file won't be found.
FEATURES_PATH = Path("eatures.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    if not FEATURES_PATH.exists():
        st.error(f"Features file not found: {FEATURES_PATH}")
        st.stop()
    return joblib.load(FEATURES_PATH)


model = load_model()
features = load_features()

# ---------------- Header ----------------
st.title("📦 Supply Chain Demand Forecasting")
st.caption("Predict future product demand using the trained model.")
st.divider()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Forecast Inputs")

    st.subheader("Sales & Promotions")
    sales_units = st.slider(
        "Current Sales Units", min_value=10, max_value=200, value=100, step=1,
        help="Units sold in the current period."
    )
    holiday_season = st.selectbox(
        "Holiday Season", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    promotion_applied = st.selectbox(
        "Promotion Applied", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    discount_percentage = st.slider(
        "Discount Percentage", min_value=0, max_value=30, value=5, step=1,
        format="%d%%"
    )

    st.divider()
    st.subheader("Market Conditions")
    competitor_price_index = st.selectbox(
        "Competitor Pricing", [0, 1],
        format_func=lambda x: "High" if x == 1 else "Low",
        help="Is competitor pricing currently high or low relative to yours?"
    )
    economic_index = st.selectbox(
        "Economic Conditions", [0, 1],
        format_func=lambda x: "Strong" if x == 1 else "Weak"
    )
    weather_impact = st.selectbox(
        "Weather Impact", [0, 1],
        format_func=lambda x: "Adverse" if x == 1 else "Normal"
    )

    st.divider()
    st.subheader("Pricing")
    price = st.slider(
        "Price", min_value=20, max_value=200, value=110, step=1
    )

    st.divider()
    st.subheader("Location & Product")
    region = st.selectbox("Region", ["Europe", "North America"])
    store_type = st.selectbox("Store Type", ["Retail", "Wholesale"])
    category = st.selectbox("Category", ["Cabinets", "Chairs", "Sofas", "Tables"])

    st.divider()
    st.subheader("Forecast Date")
    forecast_date = st.date_input("Date", value=pd.Timestamp.today().date())

    st.divider()
    predict_button = st.button(
        "🔮 Predict Demand", type="primary", use_container_width=True
    )

# ---------------- Feature preparation ----------------
date = pd.Timestamp(forecast_date)

input_data = {
    "sales_units": sales_units,
    "holiday_season": holiday_season,
    "promotion_applied": promotion_applied,
    "competitor_price_index": competitor_price_index,
    "economic_index": economic_index,
    "weather_impact": weather_impact,
    "price": price,
    "discount_percentage": discount_percentage,

    "region_Europe": 1 if region == "Europe" else 0,
    "region_North America": 1 if region == "North America" else 0,

    "store_type_Retail": 1 if store_type == "Retail" else 0,
    "store_type_Wholesale": 1 if store_type == "Wholesale" else 0,

    "category_Cabinets": 1 if category == "Cabinets" else 0,
    "category_Chairs": 1 if category == "Chairs" else 0,
    "category_Sofas": 1 if category == "Sofas" else 0,
    "category_Tables": 1 if category == "Tables" else 0,

    "year": date.year,
    "month": date.month,
    "day": date.day,
    "day_of_week": date.dayofweek,
    "week_of_year": int(date.isocalendar().week)
}

input_df = pd.DataFrame([input_data])

# Ensure exact feature order used during training
input_df = input_df.reindex(columns=features, fill_value=0)

# A friendly, plain-language version of the inputs (for display only)
readable_summary = pd.DataFrame([{
    "Forecast Date": date.strftime("%d %b %Y"),
    "Current Sales Units": sales_units,
    "Region": region,
    "Store Type": store_type,
    "Category": category,
    "Price": price,
    "Discount %": discount_percentage,
    "Holiday Season": "Yes" if holiday_season else "No",
    "Promotion Applied": "Yes" if promotion_applied else "No",
    "Competitor Pricing": "High" if competitor_price_index else "Low",
    "Economic Conditions": "Strong" if economic_index else "Weak",
    "Weather Impact": "Adverse" if weather_impact else "Normal",
}])

# ---------------- Current selection overview ----------------
with st.container(border=True):
    st.markdown("##### 📋 Current Selection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Date", date.strftime("%d %b %Y"))
    c2.metric("Sales Units", f"{sales_units:,}")
    c3.metric("Region", region)
    c4.metric("Category", category)

st.write("")

# ---------------- Prediction result ----------------
if predict_button:
    with st.spinner("Running forecast model..."):
        try:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0, prediction)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prediction = None

    if prediction is not None:
        diff = prediction - sales_units

        with st.container(border=True):
            st.markdown("### 📈 Forecast Result")

            r1, r2 = st.columns([1, 1])
            with r1:
                st.metric(
                    "Predicted Future Demand",
                    f"{prediction:,.0f} units",
                    delta=f"{diff:+,.0f} vs current sales",
                    delta_color="normal"
                )
            with r2:
                chart_df = pd.DataFrame({
                    "Units": [sales_units, prediction]
                }, index=["Current Sales", "Predicted Demand"])
                st.bar_chart(chart_df, height=180)

            # Decision-support message
            if diff > 0:
                st.warning(
                    f"📦 Expected demand is higher than current sales units. "
                    f"Approx. **{diff:,.0f}** additional units may be required."
                )
            elif diff < 0:
                st.success(
                    f"✅ Predicted demand is below current sales units "
                    f"by approximately **{abs(diff):,.0f}** units."
                )
            else:
                st.info("Predicted demand is approximately equal to current sales units.")

            with st.expander("View input summary"):
                st.dataframe(readable_summary, use_container_width=True, hide_index=True)
else:
    st.info("👈 Set your inputs in the sidebar, then click **Predict Demand** to see the forecast.")

st.divider()

st.caption(
    "Supply Chain Forecasting • Built with Python, Pandas, and XGBoost/Streamlit"
)