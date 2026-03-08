import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Predictor",
    page_icon="📦",
    layout="centered"
)

# ── Load artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load('./Models/XGBoost.joblib')
    encoder   = joblib.load('./Models/encoder.joblib')
    feat_cols = joblib.load('./Models/feature_columns.joblib')
    return model, encoder, feat_cols

model, encoder, feat_cols = load_artifacts()

# ── Label map ──────────────────────────────────────────────────
LABELS = {
    0: ('🟢 Advance Shipping',  'success'),
    1: ('🔵 Shipping On Time',  'info'),
    2: ('🔴 Late Delivery',     'error'),
    3: ('⚠️ Shipping Canceled', 'warning'),
}

# ── Constants (from training data) ─────────────────────────────
MARKETS        = ['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA']
SEGMENTS       = ['Consumer', 'Corporate', 'Home Office']
COUNTRIES      = ['Puerto Rico', 'USA']
DEPARTMENTS    = ['Apparel', 'Book Shop', 'Discs Shop', 'Fan Shop', 'Fitness',
                  'Footwear', 'Golf', 'Health and Beauty ', 'Outdoors', 'Pet Shop', 'Technology']
REGIONS        = ['Canada', 'Caribbean', 'Central Africa', 'Central America', 'Central Asia',
                  'East Africa', 'East of USA', 'Eastern Asia', 'Eastern Europe', 'North Africa',
                  'Northern Europe', 'Oceania', 'South America', 'South Asia', 'South of  USA ',
                  'Southeast Asia', 'Southern Africa', 'Southern Europe', 'US Center ',
                  'West Africa', 'West Asia', 'West of USA ', 'Western Europe']
PAYMENT_TYPES  = ['DEBIT', 'TRANSFER', 'CASH', 'PAYMENT']
SHIPPING_MODES = ['Standard Class', 'First Class', 'Second Class', 'Same Day']
DAY_ORDER      = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# ── UI ─────────────────────────────────────────────────────────
st.title("📦 Supply Chain Risk Predictor")
st.markdown("Predict shipment delivery outcome based on order details.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛒 Order Details")
    payment_type   = st.selectbox("Payment Type", PAYMENT_TYPES)
    shipping_mode  = st.selectbox("Shipping Mode", SHIPPING_MODES)
    market         = st.selectbox("Market", MARKETS)
    order_region   = st.selectbox("Order Region", REGIONS)
    order_country  = st.text_input("Order Country", value="United States")
    days_scheduled = st.slider("Days for Shipment (Scheduled)", 0, 7, 4)

with col2:
    st.subheader("👤 Customer Details")
    customer_segment = st.selectbox("Customer Segment", SEGMENTS)
    customer_country = st.selectbox("Customer Country", COUNTRIES)
    customer_state   = st.text_input("Customer State", value="CA")
    department       = st.selectbox("Department", DEPARTMENTS)
    category_name    = st.text_input("Category Name", value="Sporting Goods")
    latitude         = st.number_input("Latitude",  value=34.05)
    longitude        = st.number_input("Longitude", value=-118.24)

st.divider()
st.subheader("💰 Financial Details")
col3, col4, col5 = st.columns(3)

with col3:
    benefit_per_order   = st.number_input("Benefit per Order ($)", value=50.0)
    order_item_quantity = st.slider("Order Item Quantity", 1, 5, 1)

with col4:
    sales            = st.number_input("Sales ($)", value=200.0)
    order_item_total = st.number_input("Order Item Total ($)", value=190.0)

with col5:
    product_price = st.number_input("Product Price ($)", value=199.99)
    discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.05)
    profit_ratio  = st.slider("Profit Ratio", -2.0, 0.5, 0.2)

st.divider()
st.subheader("📅 Shipping Date")
shipping_date = st.date_input("Shipping Date", value=datetime.today())

# ── Predict ────────────────────────────────────────────────────
if st.button("🔮 Predict Delivery Outcome", use_container_width=True, type="primary"):

    # 1 — Derive date features
    dt             = pd.Timestamp(shipping_date)
    shipping_month = dt.month
    shipping_dow   = DAY_ORDER.index(dt.day_name())
    shipping_hour  = dt.hour if dt.hour != 0 else 12

    # 2 — Binary flags
    is_transfer       = 1 if payment_type == 'TRANSFER' else 0
    is_standard_class = 1 if shipping_mode == 'Standard Class' else 0
    is_second_class   = 1 if shipping_mode == 'Second Class' else 0

    # 3 — Target encode (Category Name, Customer State, Order Country)
    raw = pd.DataFrame([{
        'Category Name'  : category_name,
        'Customer State' : customer_state,
        'Order Country'  : order_country,
    }])
    encoded_cats      = encoder.transform(raw)
    cat_name_enc      = float(encoded_cats['Category Name'].iloc[0])
    cust_state_enc    = float(encoded_cats['Customer State'].iloc[0])
    order_country_enc = float(encoded_cats['Order Country'].iloc[0])

    # 4 — One-hot encode manually (match training columns exactly)
    def ohe(prefix, categories, value):
        return {f"{prefix}_{c}": int(c == value) for c in categories}

    row = {
        # Numerical
        'Days for shipment (scheduled)' : days_scheduled,
        'Benefit per order'             : benefit_per_order,
        'Category Name'                 : cat_name_enc,
        'Customer State'                : cust_state_enc,
        'Latitude'                      : latitude,
        'Longitude'                     : longitude,
        'Order Country'                 : order_country_enc,
        'Order Item Discount Rate'      : discount_rate,
        'Order Item Profit Ratio'       : profit_ratio,
        'Order Item Quantity'           : order_item_quantity,
        'Sales'                         : sales,
        'Order Item Total'              : order_item_total,
        'Product Price'                 : product_price,
        'shipping_month'                : shipping_month,
        'shipping_dayofweek'            : shipping_dow,
        'shipping_hour'                 : shipping_hour,
        # Binary flags
        'Is_Transfer'                   : is_transfer,
        'Is_Standard_Class'             : is_standard_class,
        'Is_Second_Class'               : is_second_class,
        # One-hot
        **ohe('Market', MARKETS, market),
        **ohe('Customer Segment', SEGMENTS, customer_segment),
        **ohe('Customer Country', COUNTRIES, customer_country),
        **ohe('Department Name', DEPARTMENTS, department),
        **ohe('Order Region', REGIONS, order_region),
    }

    input_df = pd.DataFrame([row])

    # 5 — Reorder columns to exactly match training
    input_df = input_df.reindex(columns=feat_cols, fill_value=0)

    # 6 — Predict (no scaling needed for XGBoost)
    pred       = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    label, lvl = LABELS[pred]

    # 7 — Display result
    st.divider()
    st.subheader("📊 Prediction Result")

    if lvl == 'success':
        st.success(f"**Predicted Outcome: {label}**")
    elif lvl == 'info':
        st.info(f"**Predicted Outcome: {label}**")
    elif lvl == 'error':
        st.error(f"**Predicted Outcome: {label}**")
    else:
        st.warning(f"**Predicted Outcome: {label}**")

    # Confidence bar chart
    proba_df = pd.DataFrame({
        'Outcome'     : [LABELS[i][0] for i in range(4)],
        'Probability' : proba
    }).set_index('Outcome')

    st.bar_chart(proba_df)

    # Detailed probabilities
    st.markdown("**Confidence scores:**")
    for i, (lbl, _) in LABELS.items():
        st.progress(float(proba[i]), text=f"{lbl}: {proba[i]*100:.1f}%")