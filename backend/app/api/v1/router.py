from fastapi import APIRouter
from app.api.v1.endpoints import analyze

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(analyze.router, tags=["analysis"])
