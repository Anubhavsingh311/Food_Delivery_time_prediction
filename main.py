from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_utils import predict_delivery_time


app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="API for predicting food delivery time",
    version="1.0.0"
)


class DeliveryInput(BaseModel):

    age: float
    rating: float

    restaurant_lat: float
    restaurant_lon: float

    delivery_lat: float
    delivery_lon: float

    vehicle_condition: int
    multiple_deliveries: float

    order_date: str

    order_hour: int
    order_minute: int

    pickup_hour: int
    pickup_minute: int

    weather: str
    traffic: str

    order_type: str
    vehicle: str

    festival: str
    city: str


@app.get("/")
def home():
    return {
        "message": "Food Delivery Time Prediction API is running"
    }


@app.post("/predict")
def predict(data: DeliveryInput):

    try:

        prediction = predict_delivery_time(
            age=data.age,
            rating=data.rating,

            restaurant_lat=data.restaurant_lat,
            restaurant_lon=data.restaurant_lon,

            delivery_lat=data.delivery_lat,
            delivery_lon=data.delivery_lon,

            vehicle_condition=data.vehicle_condition,
            multiple_deliveries=data.multiple_deliveries,

            order_date=data.order_date,

            order_hour=data.order_hour,
            order_minute=data.order_minute,

            pickup_hour=data.pickup_hour,
            pickup_minute=data.pickup_minute,

            weather=data.weather,
            traffic=data.traffic,

            order_type=data.order_type,
            vehicle=data.vehicle,

            festival=data.festival,
            city=data.city
        )

        return {
            "prediction": prediction,
            "unit": "minutes"
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )