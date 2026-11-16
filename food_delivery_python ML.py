# 1. IMPORTING LIBRARIES
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# ================================================================
# 2. PROJECT CONFIGURATION
# ================================================================

PROJECT_NAME = "Food Delivery Operations and Cancellation Bottlenecks"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

for folder in [
    DATA_DIR,
    OUTPUT_DIR,
    CHART_DIR,
    REPORT_DIR,
    MODEL_DIR
]:
    os.makedirs(folder, exist_ok=True)

DATA_PATH = os.path.join(
    DATA_DIR,
    "food_delivery_data.csv"
)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")

print("=" * 70)
print(PROJECT_NAME)
print("=" * 70)


# ================================================================
# 3. GENERATING REALISTIC FOOD DELIVERY DATA
# ================================================================

def generate_food_delivery_data(n=10000):

    cities = [
        "Kolkata",
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Pune"
    ]

    restaurants = [
        "Food Junction",
        "Spice Garden",
        "Burger House",
        "Pizza Corner",
        "Royal Biryani",
        "Urban Kitchen",
        "Tandoori Express",
        "Chinese Wok",
        "The Meal Box",
        "Fresh Bites"
    ]

    weather_conditions = [
        "Clear",
        "Rainy",
        "Stormy",
        "Cloudy"
    ]

    traffic_conditions = [
        "Low",
        "Medium",
        "High",
        "Severe"
    ]

    payment_methods = [
        "UPI",
        "Credit Card",
        "Debit Card",
        "Cash on Delivery",
        "Wallet"
    ]

    order_types = [
        "Regular",
        "Express",
        "Scheduled"
    ]

    customer_segments = [
        "New",
        "Regular",
        "Premium"
    ]

    cancellation_reasons = [
        "Customer Changed Mind",
        "Restaurant Delay",
        "Delivery Partner Unavailable",
        "Payment Failure",
        "Long Delivery Time",
        "Technical Issue"
    ]

    city = np.random.choice(cities, n)

    restaurant = np.random.choice(restaurants, n)

    weather = np.random.choice(
        weather_conditions,
        n,
        p=[0.55, 0.25, 0.08, 0.12]
    )

    traffic = np.random.choice(
        traffic_conditions,
        n,
        p=[0.25, 0.35, 0.25, 0.15]
    )

    payment = np.random.choice(payment_methods, n)

    order_type = np.random.choice(
        order_types,
        n,
        p=[0.65, 0.20, 0.15]
    )

    customer_segment = np.random.choice(
        customer_segments,
        n,
        p=[0.30, 0.50, 0.20]
    )

    hour = np.random.choice(
        np.arange(8, 24),
        n
    )

    day_of_week = np.random.randint(0, 7, n)

    order_dates = [
        datetime(2026, 1, 1) +
        timedelta(days=int(x))
        for x in np.random.randint(0, 300, n)
    ]

    order_value = np.random.uniform(100, 1500, n)

    distance = np.random.uniform(0.5, 20, n)

    preparation_time = np.random.normal(22, 8, n)

    preparation_time = np.clip(
        preparation_time,
        5,
        60
    )

    delivery_partner_wait = np.random.uniform(
        0,
        30,
        n
    )

    driver_experience = np.random.randint(
        1,
        11,
        n
    )

    available_drivers = np.random.randint(
        1,
        20,
        n
    )

    estimated_delivery_time = (
        12 +
        distance * 2.5 +
        preparation_time * 0.5
    )

    # Traffic impact
    traffic_delay = np.select(
        [
            traffic == "Low",
            traffic == "Medium",
            traffic == "High",
            traffic == "Severe"
        ],
        [0, 5, 12, 22]
    )

    # Weather impact
    weather_delay = np.select(
        [
            weather == "Clear",
            weather == "Cloudy",
            weather == "Rainy",
            weather == "Stormy"
        ],
        [0, 2, 8, 18]
    )

    actual_delivery_time = (
        estimated_delivery_time +
        traffic_delay +
        weather_delay +
        delivery_partner_wait +
        np.random.normal(0, 5, n)
    )

    actual_delivery_time = np.maximum(
        actual_delivery_time,
        5
    )

    # Create cancellation probabilities
    cancellation_score = (
        -3.2
        + (preparation_time - 22) * 0.035
        + (distance - 8) * 0.055
        + traffic_delay * 0.025
        + weather_delay * 0.025
        + delivery_partner_wait * 0.035
        + (hour >= 19) * 0.35
        + (available_drivers <= 3) * 0.75
        + (payment == "Cash on Delivery") * 0.12
        + (customer_segment == "New") * 0.30
        + (order_type == "Express") * 0.15
    )

    cancellation_probability = (
        1 / (1 + np.exp(-cancellation_score))
    )

    cancellation_probability = np.clip(
        cancellation_probability,
        0.01,
        0.90
    )

    cancelled = np.random.binomial(
        1,
        cancellation_probability
    )

    order_status = np.where(
        cancelled == 1,
        "Cancelled",
        "Delivered"
    )

    cancellation_reason = [
        np.random.choice(
            cancellation_reasons,
            p=[0.12, 0.25, 0.22, 0.08, 0.25, 0.08]
        )
        if status == 1 else "Not Applicable"
        for status in cancelled
    ]

    delivery_delay = (
        actual_delivery_time -
        estimated_delivery_time
    )

    customer_rating = np.where(
        cancelled == 1,
        np.random.randint(1, 4, n),
        np.random.randint(3, 6, n)
    )

    data = pd.DataFrame({

        "Order_ID": [
            f"FD{100001 + i}"
            for i in range(n)
        ],

        "Order_Date": order_dates,
        "City": city,
        "Restaurant": restaurant,
        "Weather": weather,
        "Traffic": traffic,
        "Payment_Method": payment,
        "Order_Type": order_type,
        "Customer_Segment": customer_segment,

        "Order_Hour": hour,
        "Day_of_Week": day_of_week,

        "Order_Value": order_value,
        "Delivery_Distance_KM": distance,

        "Preparation_Time_Min": preparation_time,
        "Delivery_Partner_Wait_Min": delivery_partner_wait,

        "Driver_Experience_Years": driver_experience,
        "Available_Drivers": available_drivers,

        "Estimated_Delivery_Time_Min": estimated_delivery_time,
        "Actual_Delivery_Time_Min": actual_delivery_time,

        "Traffic_Delay_Min": traffic_delay,
        "Weather_Delay_Min": weather_delay,

        "Delivery_Delay_Min": delivery_delay,

        "Customer_Rating": customer_rating,

        "Cancellation_Reason": cancellation_reason,

        "Order_Status": order_status,

        "Cancelled": cancelled

    })

    return data


# Generate dataset

df = generate_food_delivery_data(10000)

df.to_csv(DATA_PATH, index=False)

print("\nDataset generated successfully!")

print("Total Orders:", len(df))

print(df.head())


# ================================================================
# 4. LOADING AND UNDERSTANDING DATA
# ================================================================

df = pd.read_csv(DATA_PATH)

print("\nDATASET INFORMATION")

print(df.info())

print("\nFIRST FIVE RECORDS")

print(df.head())

print("\nSTATISTICAL SUMMARY")

print(df.describe())

print("\nDATASET SHAPE")

print(df.shape)

print("\nCOLUMN NAMES")

print(df.columns.tolist())


# ================================================================
# 5. DATA CLEANING
# ================================================================

print("\nDATA CLEANING STARTED")

print("Missing Values Before Cleaning:")

print(df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(
    subset="Order_ID",
    inplace=True
)

# Convert date column

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

# Handle missing numerical values

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:

    df[col] = df[col].fillna(
        df[col].median()
    )

# Handle categorical missing values

categorical_columns = df.select_dtypes(
    include="object"
).columns

for col in categorical_columns:

    df[col] = df[col].fillna("Unknown")

# Remove invalid records

df = df[
    (df["Order_Value"] > 0) &
    (df["Delivery_Distance_KM"] > 0) &
    (df["Preparation_Time_Min"] > 0)
]

print("\nDATA CLEANING COMPLETED")

print("Remaining Records:", len(df))

print("\nMissing Values After Cleaning:")

print(df.isnull().sum())


# ================================================================
# 6. FEATURE ENGINEERING
# ================================================================

df["Order_Month"] = df["Order_Date"].dt.month

df["Order_Day"] = df["Order_Date"].dt.day

df["Weekend"] = (
    df["Day_of_Week"] >= 5
).astype(int)

df["Peak_Hour"] = (
    (df["Order_Hour"] >= 12) &
    (df["Order_Hour"] <= 14)
    |
    (df["Order_Hour"] >= 19) &
    (df["Order_Hour"] <= 22)
).astype(int)

df["Long_Distance_Order"] = (
    df["Delivery_Distance_KM"] > 10
).astype(int)

df["Long_Preparation"] = (
    df["Preparation_Time_Min"] > 30
).astype(int)

df["Driver_Shortage"] = (
    df["Available_Drivers"] <= 3
).astype(int)

df["Late_Delivery"] = (
    df["Delivery_Delay_Min"] > 15
).astype(int)

df["High_Value_Order"] = (
    df["Order_Value"] > 1000
).astype(int)

print("\nFEATURE ENGINEERING COMPLETED")


# ================================================================
# 7. KEY PERFORMANCE INDICATORS (KPIs)
# ================================================================

total_orders = len(df)

cancelled_orders = df["Cancelled"].sum()

delivered_orders = total_orders - cancelled_orders

cancellation_rate = (
    cancelled_orders / total_orders
) * 100

delivery_rate = (
    delivered_orders / total_orders
) * 100

average_order_value = df["Order_Value"].mean()

average_delivery_time = (
    df["Actual_Delivery_Time_Min"].mean()
)

average_preparation_time = (
    df["Preparation_Time_Min"].mean()
)

average_delivery_delay = (
    df["Delivery_Delay_Min"].mean()
)

average_customer_rating = (
    df["Customer_Rating"].mean()
)

kpi_data = {

    "Total Orders": total_orders,

    "Cancelled Orders": cancelled_orders,

    "Delivered Orders": delivered_orders,

    "Cancellation Rate (%)": round(
        cancellation_rate, 2
    ),

    "Delivery Rate (%)": round(
        delivery_rate, 2
    ),

    "Average Order Value": round(
        average_order_value, 2
    ),

    "Average Delivery Time": round(
        average_delivery_time, 2
    ),

    "Average Preparation Time": round(
        average_preparation_time, 2
    ),

    "Average Delivery Delay": round(
        average_delivery_delay, 2
    ),

    "Average Customer Rating": round(
        average_customer_rating, 2
    )

}

print("\n" + "=" * 60)

print("BUSINESS PERFORMANCE KPIs")

print("=" * 60)

for key, value in kpi_data.items():

    print(f"{key}: {value}")


pd.DataFrame(
    list(kpi_data.items()),
    columns=["KPI", "Value"]
).to_csv(
    os.path.join(REPORT_DIR, "business_kpis.csv"),
    index=False
)


# ================================================================
# 8. EXPLORATORY DATA ANALYSIS
# ================================================================

print("\nEXPLORATORY DATA ANALYSIS STARTED")


# 8.1 Order Status Distribution

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Order_Status"
)

plt.title("Food Delivery Order Status Distribution")

plt.xlabel("Order Status")

plt.ylabel("Number of Orders")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "order_status.png")
)

plt.close()


# 8.2 Cancellation by City

city_cancellation = (
    df.groupby("City")["Cancelled"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))

city_cancellation.plot(
    kind="bar"
)

plt.title("Cancellation Rate by City")

plt.ylabel("Cancellation Rate (%)")

plt.xlabel("City")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "city_cancellation.png")
)

plt.close()


# 8.3 Cancellation by Restaurant

restaurant_cancellation = (
    df.groupby("Restaurant")["Cancelled"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

restaurant_cancellation.plot(
    kind="bar",
    color="coral"
)

plt.title("Restaurant-wise Cancellation Rate")

plt.xlabel("Restaurant")

plt.ylabel("Cancellation Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "restaurant_cancellation.png")
)

plt.close()


# 8.4 Cancellation by Traffic

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Traffic",
    y="Cancelled",
    estimator=lambda x: x.mean() * 100
)

plt.title("Cancellation Rate by Traffic Condition")

plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "traffic_cancellation.png")
)

plt.close()


# 8.5 Cancellation by Weather

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Weather",
    y="Cancelled",
    estimator=lambda x: x.mean() * 100
)

plt.title("Cancellation Rate by Weather")

plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "weather_cancellation.png")
)

plt.close()


# 8.6 Hourly Order Analysis

hourly_orders = (
    df.groupby("Order_Hour")
    .size()
)

plt.figure(figsize=(12, 5))

hourly_orders.plot(
    kind="line",
    marker="o"
)

plt.title("Hourly Food Delivery Orders")

plt.xlabel("Order Hour")

plt.ylabel("Total Orders")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "hourly_orders.png")
)

plt.close()


# 8.7 Hourly Cancellation Analysis

hourly_cancellation = (
    df.groupby("Order_Hour")["Cancelled"]
    .mean() * 100
)

plt.figure(figsize=(12, 5))

hourly_cancellation.plot(
    kind="line",
    marker="o"
)

plt.title("Hourly Cancellation Trend")

plt.xlabel("Order Hour")

plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "hourly_cancellation.png")
)

plt.close()


# 8.8 Delivery Distance vs Delivery Time

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Delivery_Distance_KM",
    y="Actual_Delivery_Time_Min",
    hue="Order_Status",
    alpha=0.5
)

plt.title("Delivery Distance vs Actual Delivery Time")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "distance_delivery_time.png")
)

plt.close()


# 8.9 Preparation Time Distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Preparation_Time_Min",
    bins=30,
    kde=True
)

plt.title("Restaurant Food Preparation Time Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "preparation_time.png")
)

plt.close()


# 8.10 Correlation Heatmap

correlation_data = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(15, 10))

sns.heatmap(
    correlation_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Food Delivery Operations Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "correlation_heatmap.png")
)

plt.close()


print("\nEDA CHARTS GENERATED SUCCESSFULLY")


# ================================================================
# 9. CANCELLATION BOTTLENECK ANALYSIS
# ================================================================

print("\nCANCELLATION BOTTLENECK ANALYSIS")


# Cancellation reason analysis

reason_analysis = (
    df[df["Cancelled"] == 1]
    ["Cancellation_Reason"]
    .value_counts()
    .reset_index()
)

reason_analysis.columns = [
    "Cancellation_Reason",
    "Number_of_Cancellations"
]

reason_analysis["Percentage"] = (
    reason_analysis["Number_of_Cancellations"] /
    cancelled_orders
) * 100

reason_analysis.to_csv(
    os.path.join(
        REPORT_DIR,
        "cancellation_reason_analysis.csv"
    ),
    index=False
)

print("\nCancellation Reasons:")

print(reason_analysis)


# Delivery partner shortage analysis

driver_analysis = (
    df.groupby("Driver_Shortage")["Cancelled"]
    .mean()
    .mul(100)
    .reset_index()
)

driver_analysis["Driver_Shortage"] = (
    driver_analysis["Driver_Shortage"]
    .map({
        0: "Normal Availability",
        1: "Driver Shortage"
    })
)

print("\nDriver Availability Analysis:")

print(driver_analysis)


# Long preparation analysis

preparation_analysis = (
    df.groupby("Long_Preparation")["Cancelled"]
    .mean()
    .mul(100)
    .reset_index()
)

preparation_analysis["Long_Preparation"] = (
    preparation_analysis["Long_Preparation"]
    .map({
        0: "Normal Preparation",
        1: "Long Preparation"
    })
)

print("\nPreparation Bottleneck Analysis:")

print(preparation_analysis)


# Save operational bottleneck reports

driver_analysis.to_csv(
    os.path.join(
        REPORT_DIR,
        "driver_shortage_analysis.csv"
    ),
    index=False
)

preparation_analysis.to_csv(
    os.path.join(
        REPORT_DIR,
        "preparation_bottleneck_analysis.csv"
    ),
    index=False
)


# ================================================================
# 10. MACHINE LEARNING - CANCELLATION PREDICTION
# ================================================================

print("\n" + "=" * 70)

print("MACHINE LEARNING: CANCELLATION PREDICTION")

print("=" * 70)


# Target variable

target = "Cancelled"


# Features available for prediction at order placement.
# Actual delivery time, actual delay, rating, cancellation reason
# and order status are deliberately excluded to prevent leakage.

features = [

    "City",
    "Restaurant",
    "Weather",
    "Traffic",
    "Payment_Method",
    "Order_Type",
    "Customer_Segment",

    "Order_Hour",
    "Day_of_Week",

    "Order_Value",
    "Delivery_Distance_KM",

    "Preparation_Time_Min",
    "Delivery_Partner_Wait_Min",

    "Driver_Experience_Years",
    "Available_Drivers",

    "Estimated_Delivery_Time_Min",
    "Traffic_Delay_Min",
    "Weather_Delay_Min",

    "Weekend",
    "Peak_Hour",
    "Long_Distance_Order",
    "Long_Preparation",
    "Driver_Shortage",
    "High_Value_Order"

]


X = df[features]

y = df[target]


# Identify feature types

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


# Preprocessing pipeline

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )

    ]

)


# Split data

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)


print("\nTraining Records:", len(X_train))

print("Testing Records:", len(X_test))


# ================================================================
# 11. MODEL 1 - LOGISTIC REGRESSION
# ================================================================

logistic_model = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )

    ]

)


print("\nTraining Logistic Regression Model...")

logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(
    X_test
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test)[:, 1]
)


# ================================================================
# 12. MODEL 2 - RANDOM FOREST
# ================================================================

random_forest_model = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(

                n_estimators=300,

                max_depth=15,

                min_samples_split=5,

                min_samples_leaf=2,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1

            )
        )

    ]

)


print("\nTraining Random Forest Model...")

random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = random_forest_model.predict(
    X_test
)

rf_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# ================================================================
# 13. MODEL EVALUATION
# ================================================================

def evaluate_model(
    model_name,
    actual,
    predicted,
    probabilities
):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    auc = roc_auc_score(
        actual,
        probabilities
    )

    return {

        "Model": model_name,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1_Score": round(f1, 4),

        "ROC_AUC": round(auc, 4)

    }


logistic_results = evaluate_model(

    "Logistic Regression",

    y_test,

    logistic_predictions,

    logistic_probabilities

)


rf_results = evaluate_model(

    "Random Forest",

    y_test,

    rf_predictions,

    rf_probabilities

)


model_comparison = pd.DataFrame([

    logistic_results,

    rf_results

])


print("\nMODEL PERFORMANCE COMPARISON")

print(model_comparison)


model_comparison.to_csv(

    os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    ),

    index=False

)


# Classification report

print("\nRANDOM FOREST CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ================================================================
# 14. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Delivered", "Cancelled"],
    yticklabels=["Delivered", "Cancelled"]
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "confusion_matrix.png"
    )
)

plt.close()


# ================================================================
# 15. ROC CURVE
# ================================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest AUC = {rf_results['ROC_AUC']}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Cancellation Prediction")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "roc_curve.png"
    )
)

plt.close()


# ================================================================
# 16. FEATURE IMPORTANCE
# ================================================================

rf_classifier = (
    random_forest_model.named_steps["classifier"]
)

encoded_features = (
    random_forest_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance = pd.DataFrame({

    "Feature": encoded_features,

    "Importance": rf_classifier.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTOP 15 IMPORTANT FEATURES")

print(
    feature_importance.head(15)
)


feature_importance.to_csv(

    os.path.join(
        REPORT_DIR,
        "feature_importance.csv"
    ),

    index=False

)


plt.figure(figsize=(12, 8))

sns.barplot(

    data=feature_importance.head(15),

    x="Importance",

    y="Feature"

)

plt.title("Top 15 Cancellation Prediction Features")

plt.tight_layout()

plt.savefig(

    os.path.join(
        CHART_DIR,
        "feature_importance.png"
    )

)

plt.close()


# ================================================================
# 17. CANCELLATION RISK PREDICTION
# ================================================================

prediction_data = X_test.copy()

prediction_data["Actual_Cancellation"] = (
    y_test.values
)

prediction_data["Predicted_Cancellation"] = (
    rf_predictions
)

prediction_data["Cancellation_Probability"] = (
    rf_probabilities
)

prediction_data["Risk_Level"] = pd.cut(

    rf_probabilities,

    bins=[-0.001, 0.30, 0.60, 1.001],

    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]

)

print("\nCANCELLATION RISK DISTRIBUTION")

print(
    prediction_data["Risk_Level"]
    .value_counts()
)


prediction_data.to_csv(

    os.path.join(
        REPORT_DIR,
        "cancellation_risk_predictions.csv"
    ),

    index=False

)


# ================================================================
# 18. K-MEANS CUSTOMER AND OPERATIONAL SEGMENTATION
# ================================================================

print("\nK-MEANS CLUSTERING STARTED")


cluster_features = [

    "Order_Value",

    "Delivery_Distance_KM",

    "Preparation_Time_Min",

    "Actual_Delivery_Time_Min",

    "Delivery_Partner_Wait_Min",

    "Customer_Rating"

]


cluster_data = df[cluster_features].copy()


cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)


kmeans = KMeans(

    n_clusters=4,

    random_state=RANDOM_STATE,

    n_init=10

)


df["Operational_Cluster"] = (
    kmeans.fit_predict(cluster_scaled)
)


cluster_summary = (

    df.groupby("Operational_Cluster")

    .agg({

        "Order_Value": "mean",

        "Delivery_Distance_KM": "mean",

        "Preparation_Time_Min": "mean",

        "Actual_Delivery_Time_Min": "mean",

        "Cancelled": "mean",

        "Customer_Rating": "mean"

    })

    .reset_index()

)


cluster_summary["Cancellation_Rate_Percent"] = (

    cluster_summary["Cancelled"] * 100

)


cluster_summary.drop(

    columns=["Cancelled"],

    inplace=True

)


print("\nOPERATIONAL CLUSTER SUMMARY")

print(cluster_summary)


cluster_summary.to_csv(

    os.path.join(
        REPORT_DIR,
        "operational_cluster_analysis.csv"
    ),

    index=False

)


# Cluster visualization

plt.figure(figsize=(10, 7))

sns.scatterplot(

    data=df,

    x="Delivery_Distance_KM",

    y="Actual_Delivery_Time_Min",

    hue="Operational_Cluster",

    palette="viridis",

    alpha=0.6

)

plt.title("Food Delivery Operational Segmentation")

plt.tight_layout()

plt.savefig(

    os.path.join(
        CHART_DIR,
        "operational_clusters.png"
    )

)

plt.close()


# ================================================================
# 19. RESTAURANT PERFORMANCE ANALYSIS
# ================================================================

restaurant_performance = (

    df.groupby("Restaurant")

    .agg({

        "Order_ID": "count",

        "Cancelled": "mean",

        "Preparation_Time_Min": "mean",

        "Actual_Delivery_Time_Min": "mean",

        "Customer_Rating": "mean",

        "Order_Value": "mean"

    })

    .reset_index()

)


restaurant_performance.rename(

    columns={

        "Order_ID": "Total_Orders",

        "Cancelled": "Cancellation_Rate",

        "Preparation_Time_Min": "Avg_Preparation_Time",

        "Actual_Delivery_Time_Min": "Avg_Delivery_Time",

        "Customer_Rating": "Avg_Customer_Rating",

        "Order_Value": "Avg_Order_Value"

    },

    inplace=True

)


restaurant_performance["Cancellation_Rate"] *= 100


restaurant_performance.to_csv(

    os.path.join(
        REPORT_DIR,
        "restaurant_performance.csv"
    ),

    index=False

)


print("\nRESTAURANT PERFORMANCE")

print(
    restaurant_performance.sort_values(
        by="Cancellation_Rate",
        ascending=False
    )
)

# ================================================================
# 20. CITY-WISE OPERATIONAL PERFORMANCE
# ================================================================

city_performance = (

    df.groupby("City")

    .agg({

        "Order_ID": "count",

        "Cancelled": "mean",

        "Actual_Delivery_Time_Min": "mean",

        "Delivery_Delay_Min": "mean",

        "Customer_Rating": "mean"

    })

    .reset_index()

)

# Rename columns FIRST

city_performance.rename(

    columns={

        "Order_ID": "Total_Orders",

        "Cancelled": "Cancellation_Rate",

        "Actual_Delivery_Time_Min": "Avg_Delivery_Time",

        "Delivery_Delay_Min": "Avg_Delivery_Delay",

        "Customer_Rating": "Avg_Customer_Rating"

    },

    inplace=True

)

# Convert cancellation rate into percentage

city_performance["Cancellation_Rate"] = (

    city_performance["Cancellation_Rate"] * 100

).round(2)


# Round other metrics

city_performance["Avg_Delivery_Time"] = (

    city_performance["Avg_Delivery_Time"].round(2)

)

city_performance["Avg_Delivery_Delay"] = (

    city_performance["Avg_Delivery_Delay"].round(2)

)

city_performance["Avg_Customer_Rating"] = (

    city_performance["Avg_Customer_Rating"].round(2)

)


# Display results

print("\nCITY-WISE OPERATIONAL PERFORMANCE")

print(city_performance.to_string(index=False))


# Save report

city_performance.to_csv(

    os.path.join(

        REPORT_DIR,

        "city_performance.csv"

    ),

    index=False

)

print("\nCity performance report saved successfully!")



# ================================================================
# 21. DELIVERY TIME PERFORMANCE
# ================================================================

df["Delivery_Performance"] = np.select(

    [

        df["Delivery_Delay_Min"] <= 0,

        df["Delivery_Delay_Min"] <= 10,

        df["Delivery_Delay_Min"] <= 20

    ],

    [

        "On Time or Early",

        "Slight Delay",

        "Moderate Delay"

    ],

    default="High Delay"

)


delivery_performance = (

    df.groupby("Delivery_Performance")

    .size()

    .reset_index(name="Number_of_Orders")

)


delivery_performance.to_csv(

    os.path.join(
        REPORT_DIR,
        "delivery_time_performance.csv"
    ),

    index=False

)


# ================================================================
# 22. BUSINESS INSIGHTS GENERATION
# ================================================================

print("\n" + "=" * 70)

print("AUTOMATED BUSINESS INSIGHTS")

print("=" * 70)


worst_city = city_performance.sort_values(

    "Cancellation_Rate",

    ascending=False

).iloc[0]


worst_restaurant = restaurant_performance.sort_values(

    "Cancellation_Rate",

    ascending=False

).iloc[0]


top_reason = reason_analysis.iloc[0]


print(
    f"\n1. Overall cancellation rate: "
    f"{cancellation_rate:.2f}%"
)


print(

    f"\n2. City with highest cancellation rate: "

    f"{worst_city['City']} "

    f"({worst_city['Cancellation_Rate']:.2f}%)"

)


print(

    f"\n3. Restaurant with highest cancellation rate: "

    f"{worst_restaurant['Restaurant']} "

    f"({worst_restaurant['Cancellation_Rate']:.2f}%)"

)


print(

    f"\n4. Most common cancellation reason: "

    f"{top_reason['Cancellation_Reason']}"

)


print(

    f"\n5. Average delivery time: "

    f"{average_delivery_time:.2f} minutes"

)


print(

    f"\n6. Average preparation time: "

    f"{average_preparation_time:.2f} minutes"

)


print(

    f"\n7. Average customer rating: "

    f"{average_customer_rating:.2f}/5"

)


# ================================================================
# 23. BUSINESS RECOMMENDATIONS
# ================================================================

recommendations = [

    {

        "Bottleneck": "Restaurant Preparation Delay",

        "Recommendation":
        "Optimize kitchen workflows and monitor restaurant preparation time."

    },

    {

        "Bottleneck": "Delivery Partner Shortage",

        "Recommendation":
        "Increase driver availability during peak ordering hours."

    },

    {

        "Bottleneck": "Traffic Congestion",

        "Recommendation":
        "Introduce route optimization and better delivery time estimation."

    },

    {

        "Bottleneck": "Weather Conditions",

        "Recommendation":
        "Adjust delivery estimates during rain and severe weather."

    },

    {

        "Bottleneck": "Long Delivery Distance",

        "Recommendation":
        "Improve restaurant allocation and prioritize nearby restaurants."

    },

    {

        "Bottleneck": "High Cancellation Risk",

        "Recommendation":
        "Use machine learning predictions to identify risky orders early."

    },

    {

        "Bottleneck": "Poor Customer Experience",

        "Recommendation":
        "Improve order tracking, communication, and delivery transparency."

    }

]


recommendation_df = pd.DataFrame(
    recommendations
)


recommendation_df.to_csv(

    os.path.join(
        REPORT_DIR,
        "business_recommendations.csv"
    ),

    index=False

)


print("\nBUSINESS RECOMMENDATIONS")

for index, row in recommendation_df.iterrows():

    print(
        f"\n{index + 1}. {row['Bottleneck']}"
    )

    print(
        row["Recommendation"]
    )


# ================================================================
# 24. SAVING MACHINE LEARNING MODELS
# ================================================================

joblib.dump(

    logistic_model,

    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )

)


joblib.dump(

    random_forest_model,

    os.path.join(
        MODEL_DIR,
        "random_forest_model.pkl"
    )

)


joblib.dump(

    kmeans,

    os.path.join(
        MODEL_DIR,
        "kmeans_model.pkl"
    )

)


joblib.dump(

    cluster_scaler,

    os.path.join(
        MODEL_DIR,
        "cluster_scaler.pkl"
    )

)


# ================================================================
# 25. FINAL PROJECT SUMMARY
# ================================================================

print("\n" + "=" * 70)

print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")

print("=" * 70)


print("\nPROJECT:", PROJECT_NAME)

print("\nTOTAL ORDERS:", total_orders)

print("CANCELLED ORDERS:", cancelled_orders)

print("DELIVERED ORDERS:", delivered_orders)

print(
    "CANCELLATION RATE:",
    round(cancellation_rate, 2),
    "%"
)

print(
    "LOGISTIC REGRESSION ACCURACY:",
    logistic_results["Accuracy"]
)

print(
    "RANDOM FOREST ACCURACY:",
    rf_results["Accuracy"]
)

print("\nGENERATED OUTPUTS:")

print("1. Cleaned food delivery dataset")

print("2. Business KPI report")

print("3. Cancellation analysis")

print("4. Restaurant performance report")

print("5. City performance report")

print("6. Delivery performance report")

print("7. Machine learning model comparison")

print("8. Cancellation risk predictions")

print("9. Feature importance analysis")

print("10. Operational clustering")

print("11. Business recommendations")

print("12. Visualization charts")

print("\nOUTPUT LOCATION:")

print(OUTPUT_DIR)

print("\nTHANK YOU!")

print("=" * 70)
