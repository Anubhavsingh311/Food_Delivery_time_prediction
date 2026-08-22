import os
import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Model paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_lgbm.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "model",
    "preprocessor.pkl"
)


# --------------------------------------------------
# Load trained model and preprocessor
# --------------------------------------------------

best_lgbm = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# Haversine distance
# --------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# --------------------------------------------------
# Time period
# --------------------------------------------------

def get_time_period(hour):

    if pd.isna(hour):
        return "Unknown"

    if 6 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    else:
        return "Night"


# --------------------------------------------------
# Reusable prediction function
# --------------------------------------------------

def predict_delivery_time(
    age,
    rating,
    restaurant_lat,
    restaurant_lon,
    delivery_lat,
    delivery_lon,
    vehicle_condition,
    multiple_deliveries,
    order_date,
    order_hour,
    order_minute,
    pickup_hour,
    pickup_minute,
    weather,
    traffic,
    order_type,
    vehicle,
    festival,
    city
):

    # -----------------------------------------------
    # 1. Convert order date
    # -----------------------------------------------

    order_date = pd.to_datetime(
        order_date,
        format="%d-%m-%Y"
    )

    order_day = order_date.day
    order_month = order_date.month
    order_day_of_week = order_date.dayofweek

    is_weekend = int(
        order_day_of_week in [5, 6]
    )


    # -----------------------------------------------
    # 2. Calculate distance
    # -----------------------------------------------

    distance_km = haversine_distance(
        restaurant_lat,
        restaurant_lon,
        delivery_lat,
        delivery_lon
    )


    # -----------------------------------------------
    # 3. Calculate pickup delay
    # -----------------------------------------------

    order_total_minutes = (
        order_hour * 60
        + order_minute
    )

    pickup_total_minutes = (
        pickup_hour * 60
        + pickup_minute
    )

    pickup_delay = (
        pickup_total_minutes
        - order_total_minutes
    )


    # Handle midnight rollover
    if pickup_delay < 0:
        pickup_delay += 24 * 60


    # -----------------------------------------------
    # 4. Missing-value indicators
    # -----------------------------------------------

    time_order_missing = 0
    pickup_delay_missing = 0


    # -----------------------------------------------
    # 5. Time periods
    # -----------------------------------------------

    time_period = get_time_period(
        order_hour
    )

    pickup_time_period = get_time_period(
        pickup_hour
    )


    # -----------------------------------------------
    # 6. Create DataFrame
    # -----------------------------------------------

    new_order = pd.DataFrame({

        "Delivery_person_Age": [age],

        "Delivery_person_Ratings": [rating],

        "Restaurant_latitude": [restaurant_lat],

        "Restaurant_longitude": [restaurant_lon],

        "Delivery_location_latitude": [delivery_lat],

        "Delivery_location_longitude": [delivery_lon],

        "Vehicle_condition": [vehicle_condition],

        "multiple_deliveries": [multiple_deliveries],

        "Order_Hour": [order_hour],

        "Order_Minute": [order_minute],

        "Order_Day": [order_day],

        "Order_Month": [order_month],

        "Order_DayOfWeek": [order_day_of_week],

        "Is_Weekend": [is_weekend],

        "Pickup_Hour": [pickup_hour],

        "Pickup_Minute": [pickup_minute],

        "Time_Order_Missing": [time_order_missing],

        "Pickup_Delay_Missing": [pickup_delay_missing],

        "Weatherconditions": [weather],

        "Road_traffic_density": [traffic],

        "Type_of_order": [order_type],

        "Type_of_vehicle": [vehicle],

        "Festival": [festival],

        "City": [city],

        "distance_km": [distance_km],

        "Pickup_Delay_Min": [pickup_delay],

        "Time_Period": [time_period],

        "Pickup_Time_Period": [pickup_time_period]
    })


    # -----------------------------------------------
    # 7. Apply the trained preprocessor
    # -----------------------------------------------

    new_order_encoded = preprocessor.transform(
        new_order
    )


    # -----------------------------------------------
    # 8. Make prediction
    # -----------------------------------------------

    prediction = best_lgbm.predict(
        new_order_encoded
    )


    # -----------------------------------------------
    # 9. Return prediction as Python float
    # -----------------------------------------------

    return float(
        np.asarray(prediction).reshape(-1)[0]
    )