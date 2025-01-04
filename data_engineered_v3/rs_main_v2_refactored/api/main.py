from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from generate_recommendations import RecommendationGenerator
import torch

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add your Windows IP if needed
        "http://192.168.1.x:3000"  # Replace x with your actual IP
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data once when starting the server
try:
    model_path = '../checkpoints/best_model.pth'
    catalog_data = pd.read_csv('../../data/o2_data.csv')
    recommender = RecommendationGenerator(model_path, catalog_data)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class UserInput(BaseModel):
    user_id: str
    age: int
    gender: str
    genre: str
    music: str

@app.post("/recommendations/")
async def get_recommendations(user_input: UserInput):
    try:
        user_info = {
            'user_id': user_input.user_id,
            'age': user_input.age,
            'gender': user_input.gender,
            'genre': user_input.genre,
            'music': user_input.music
        }
        
        recommendations = recommender.generate_recommendations(user_info, n_recommendations=10)
        
        return {
            "status": "success",
            "recommendations": recommendations.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
