CREATE DATABASE food_delivery_analytics;
USE food_delivery_analytics;
USE food_delivery_analytics;

CREATE TABLE food_delivery_orders (
    order_id VARCHAR(30) PRIMARY KEY,
    order_date DATETIME,
    city VARCHAR(100),
    restaurant VARCHAR(150),
    weather VARCHAR(50),
    traffic VARCHAR(50),
    payment_method VARCHAR(50),
    order_type VARCHAR(50),
    customer_segment VARCHAR(50),
    order_hour INT,
    day_of_week INT,
    order_value DECIMAL(12,2),
    delivery_distance_km DECIMAL(10,2),
    preparation_time_min DECIMAL(10,2),
    delivery_partner_wait_min DECIMAL(10,2),
    driver_experience_years INT,
    available_drivers INT,
    estimated_delivery_time_min DECIMAL(10,2),
    actual_delivery_time_min DECIMAL(10,2),
    traffic_delay_min DECIMAL(10,2),
    weather_delay_min DECIMAL(10,2),
    delivery_delay_min DECIMAL(10,2),
    customer_rating INT,
    cancellation_reason VARCHAR(150),
    order_status VARCHAR(50),
    cancelled TINYINT
);
USE food_delivery_analytics;

SELECT COUNT(*) AS total_orders
FROM food_delivery_orders;

SELECT *
FROM food_delivery_orders
LIMIT 50;