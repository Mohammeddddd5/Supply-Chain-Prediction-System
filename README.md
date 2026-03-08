# 📦 Supply Chain Shipment Delay Risk Classifier
A full end-to-end machine learning project that predicts shipment delivery outcomes for supply chain orders using the DataCo Smart Supply Chain dataset. The project covers data cleaning, EDA, unsupervised learning, supervised modeling, and a Streamlit deployment app.

## 🎯 Problem Statement
Given an order's details (payment type, shipping mode, market, customer info, financials, and shipping date), predict one of four delivery outcomes:
ClassLabel0🟢 Advance Shipping1🔵 Shipping On Time2🔴 Late Delivery3⚠️ Shipping Canceled

## 📊 Dataset

Source: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
Size: 180,519 rows × 53 columns
Class distribution (imbalanced):

Late delivery: 54.8%
Advance shipping: 23.0%
Shipping on time: 17.8%
Shipping canceled: 4.3%


## 🗂️ Project Structure
├── Datasets/
│   └── DataCoSupplyChainDataset.csv
├── Models/
│   ├── XGBoost.joblib
│   ├── encoder.joblib
│   └── feature_columns.joblib
├── SupplyChain.ipynb          # Full pipeline notebook
├── app.py                     # Streamlit prediction app
└── README.md

## 🔧 Pipeline Overview
### 1. Data Cleaning

Dropped PII columns (emails, passwords, names, IDs)
Dropped leakage columns (Days for shipping (real), Late_delivery_risk, Delivery Status used only as target)
Removed redundant columns (Order Profit Per Order, Order Item Product Price, Sales per customer)
Fixed misencoded customer states (zip codes stored as state values)
Standardized Customer Country encoding (EE. UU. → USA)
Filtered extreme outliers: Order Item Discount > 450, Benefit per order < -2900, Sales > 550
Final shape after cleaning: 180,031 rows × 30 columns

### 2. Exploratory Data Analysis

Class distribution analysis
Correlation heatmap of numerical features
Countplots by: Shipping Mode, Market, Customer Segment, Customer Country, Department, Order Region
World choropleth map of late delivery rates by country
Time series trends: by year, month, day of week, hour
Key finding: TRANSFER payment type appears exclusively in Shipping Canceled orders

### 3. Feature Engineering

Binary flags: Is_Transfer, Is_Standard_Class, Is_Second_Class
Datetime features: shipping_month, shipping_dayofweek, shipping_hour
shipping_year dropped due to temporal CV instability (2018 severely underrepresented)

### 4. Encoding

One-hot encoding: Market, Customer Segment, Customer Country, Department Name, Order Region
Target encoding: Category Name, Customer State, Order Country (high cardinality)
Ordinal encoding: shipping_dayofweek (Monday=0 → Sunday=6)
Final feature matrix: 64 features

### 5. Unsupervised Learning

StandardScaler applied for unsupervised pipeline only
PCA: 40 components → 88.7% variance explained
UMAP: 2D projection (n_neighbors=50, min_dist=0.1) for visualization
KMeans: k=4 clusters fitted on PCA components
Isolation Forest: Anomaly detection (contamination=5%)

### 6. Supervised Modeling

Class Imbalance Handling
Used a combined SMOTE + RandomUnderSampler pipeline:

Oversampled minority classes (0, 1, 3) to 60,000 samples each
Undersampled majority class (2 — Late Delivery) to 60,000 samples

### 7. Models Trained

ModelCV Macro F1Test AccuracyTest Macro F1LightGBM0.870.720.60XGBoost0.900.750.65
Best Model — XGBoost
colsample_bytree = 0.8     gamma            = 0.1
learning_rate    = 0.15    max_depth        = 13
min_child_weight = 3       n_estimators     = 500
reg_alpha        = 0       reg_lambda       = 1.5
subsample        = 0.7
Final Test Results
                   precision    recall  f1-score   support
 Advance shipping       0.64      0.73      0.68      8,294
 Shipping on time       0.79      0.42      0.55      6,420
    Late delivery       0.80      0.90      0.85     19,746
Shipping canceled       0.76      0.39      0.51      1,547

         accuracy                           0.75     36,007
        macro avg       0.75      0.61      0.65     36,007
     weighted avg       0.76      0.75      0.74     36,007

Note: Supply chain delay prediction is inherently difficult due to external factors not captured in the data (weather, port congestion, carrier issues). Industry benchmarks report 70–80% accuracy on similar datasets. This model matches or exceeds human-level logistics prediction performance (~70–75%).

## 📈 Key Findings

Late delivery is by far the most predictable class (F1: 0.85) due to its dominance in the dataset
TRANSFER payment type is a strong signal for shipping cancellation — it appears exclusively in canceled orders
Standard Class shipping mode has the highest association with both advance and late delivery
UMAP visualization shows heavy class overlap, confirming the problem's inherent difficulty
Temporal data (2015–2018) shows 2018 is severely underrepresented, causing instability in time-based cross-validation folds
