# 🍔 Food Delivery Operations and Cancellation Bottlenecks

An end-to-end data analytics project using **Python, MySQL, and Power BI** to analyze food delivery operations, identify order cancellation patterns, investigate delivery bottlenecks, and generate actionable business insights.

## 📌 Project Overview

The **Food Delivery Operations and Cancellation Bottlenecks** project focuses on analyzing food delivery data to understand operational performance, customer behavior, order cancellations, delivery delays, restaurant preparation issues, and driver availability.

The project combines Python-based data analysis and machine learning, MySQL-based database analysis, and Power BI dashboards to transform raw data into meaningful business insights.

The dataset contains 10,000 synthetically generated food delivery records across multiple Indian cities.

[food_delivery_analytics.pdf](https://github.com/user-attachments/files/34907825/food_delivery_analytics.pdf)


## 🎯 Project Objectives

- Analyze overall food delivery operational performance.
- Identify major reasons behind order cancellations.
- Compare cancellation rates across cities and restaurants.
- Analyze delivery delays and preparation bottlenecks.
- Evaluate driver availability and traffic conditions.
- Understand customer segments and payment preferences.
- Build machine learning models to predict order cancellations.
- Segment operational orders using clustering.
- Create interactive Power BI dashboards for business reporting.
- Provide data-driven recommendations to improve delivery operations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data cleaning, analysis, visualization, and machine learning |
| Pandas & NumPy | Data manipulation and processing |
| Matplotlib & Seaborn | Exploratory data visualization |
| Scikit-learn | Machine learning and clustering |
| MySQL | Database management and SQL analysis |
| Power BI | Interactive dashboards and business intelligence |
| Joblib | Saving trained machine learning models |
| Git & GitHub | Version control and project hosting |

## 📂 Dataset Description

The project uses a synthetic food delivery dataset containing 10,000 records.

The dataset includes information such as:

- Order ID
- Customer ID
- Restaurant Name
- City
- Order Date and Time
- Order Value
- Order Type
- Payment Method
- Customer Segment
- Weather Condition
- Traffic Condition
- Delivery Distance
- Estimated Delivery Time
- Actual Delivery Time
- Preparation Time
- Driver Availability
- Driver Experience
- Partner Waiting Time
- Order Status
- Cancellation Reason
- Customer Rating

## 🐍 1. Python Data Analysis and Machine Learning

Python is used for data generation, preprocessing, exploratory data analysis, machine learning, and business insight generation.

### Data Cleaning and Preprocessing
- Duplicate record identification and removal.
- Missing value treatment.
- Date and time conversion.
- Invalid value handling.
- Numerical and categorical data preprocessing.

### Feature Engineering

Created additional analytical features:

- Order Month
- Order Day
- Weekend
- Peak Hour
- Long Distance Order
- Long Preparation
- Driver Shortage
- Late Delivery
- High Value Order

### Exploratory Data Analysis (EDA)

Analyzed operational patterns through visualizations:

- Order Status Distribution
- Cancellation by City
- Cancellation by Restaurant
- Cancellation by Traffic Condition
- Cancellation by Weather Condition
- Hourly Order Trends
- Hourly Cancellation Trends
- Distance vs Delivery Time
- Preparation Time Distribution
- Correlation Heatmap

### Machine Learning

Developed machine learning models to predict order cancellations.

**Models Used:**
- Logistic Regression
- Random Forest Classifier

**Model Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

### Customer and Operational Segmentation

Applied K-Means Clustering to identify operational order segments using features such as order value, delivery distance, preparation time, delivery time, partner waiting time, and customer rating.

### Cancellation Risk Prediction

Categorized predicted cancellation probabilities into:

- Low Risk
- Medium Risk
- High Risk

## 🗄️ 2. MySQL Database Analysis

MySQL is used to store and analyze food delivery order records through structured SQL queries.

**Database Name:** `food_delivery_analytics`

**Table Name:** `food_delivery_orders`

### SQL Analysis Performed

- Total Order Analysis
- Revenue Analysis by Order Status
- Payment Method Performance
- Customer Segment Analysis
- Order Type Analysis
- Weekday Performance Analysis
- Delivery Distance Categories
- Driver Experience Analysis
- Estimated vs Actual Delivery Time
- Monthly Revenue and Cancellation Trends
- Cancellation Reasons by Weather
- Restaurant Cancellation Ranking
- Time Slot Performance
- Driver Availability Analysis
- Month-over-Month Comparison

### SQL Concepts Used

- SELECT, WHERE, ORDER BY
- GROUP BY and HAVING
- Aggregate Functions
- CASE Statements
- Common Table Expressions (CTEs)
- Window Functions
- RANK()
- LAG()
- PARTITION BY
- Date and Time Functions
- Conditional Aggregation

## 📊 3. Power BI Dashboard and Business Intelligence

Power BI is used to present food delivery performance through interactive dashboards and visual reports.

The dashboard provides a business-focused view of operational performance, order trends, cancellations, delivery efficiency, and customer behavior.

### Dashboard Analysis Areas

**1. Overall Business Performance**
- Total Orders
- Delivered Orders
- Cancelled Orders
- Cancellation Rate
- Delivery Rate
- Average Order Value

**2. Order and Cancellation Analysis**
- Order status distribution
- Cancellation trends
- City-wise cancellation performance
- Restaurant-wise cancellation analysis
- Cancellation reasons

**3. Delivery Operations**
- Estimated vs Actual Delivery Time
- Delivery delay analysis
- Delivery distance analysis
- Driver availability
- Traffic and weather impact

**4. Customer and Payment Analysis**
- Customer segment performance
- Payment method distribution
- Order type analysis
- Customer ratings

**5. Time-Based Analysis**
- Monthly order trends
- Monthly cancellation trends
- Weekday performance
- Hourly order patterns
- Peak-hour operational performance

### Power BI Capabilities

- Interactive data visualizations
- KPI reporting
- Slicers and filters
- Comparative performance analysis
- Business performance monitoring
- Operational insights through dashboards

## 📈 4. Business Insights and Recommendations

The project focuses on identifying operational challenges and recommending improvements.

| Operational Challenge | Business Recommendation |
|---|---|
| High order cancellations | Identify cancellation patterns and improve order confirmation |
| Driver shortages | Increase driver availability during peak hours |
| Long preparation time | Optimize restaurant preparation workflows |
| Delivery delays | Improve route planning and delivery estimates |
| Traffic and weather impact | Adjust delivery time estimates based on conditions |
| High-risk orders | Use machine learning cancellation predictions |
| Customer dissatisfaction | Improve order tracking and communication |

## 📁 Project Outputs

The Python implementation generates analytical reports, charts, and machine learning model files.

**Generated reports include:**
- Business KPIs
- Cancellation Reason Analysis
- Driver Shortage Analysis
- Preparation Bottleneck Analysis
- Model Comparison
- Feature Importance
- Cancellation Risk Predictions
- Operational Cluster Analysis
- Business Recommendations

**Saved machine learning models:**
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `kmeans_model.pkl`
- `cluster_scaler.pkl`

## 📂 Suggested Project Structure

```text
Food-Delivery-Operations-and-Cancellation-Bottlenecks/
│
├── data/
│   └── food_delivery_data.csv
│
├── python/
│   └── food_delivery_python_ML.py
│
├── mysql/
│   └── food_delivery_analysis.sql
│
├── powerbi/
│   └── food_delivery_dashboard.pbix
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── models/
│
├── README.md
└── requirements.txt
```

*This is a suggested repository structure. Adjust filenames and folders to match the files you actually upload.*

## 🚀 Project Workflow

1. Generate the food delivery dataset.
2. Import and preprocess the data using Python.
3. Perform exploratory data analysis.
4. Analyze operational performance using MySQL.
5. Build machine learning models for cancellation prediction.
6. Perform operational segmentation using K-Means.
7. Develop interactive Power BI dashboards.
8. Interpret findings and generate business recommendations.

## 💡 Future Enhancements

- Real-time food delivery performance monitoring.
- Integration with live delivery tracking systems.
- Advanced cancellation prediction models.
- Automated dashboard refresh.
- Real-world food delivery dataset integration.
- Advanced customer behavior analysis.

## 👨‍💻 Author

**Shadmaan Ahmed**

Data Analytics | Python | SQL | Power BI | Machine Learning

---

⭐ If you find this project useful, consider giving the repository a star!

**Note:** The project dataset is synthetically generated for analytical and educational purposes. Machine learning performance metrics should be added only after running the models and obtaining actual results.
