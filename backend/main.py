from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.routes.predict import router as predict_router
from backend.routes.stocks import router as stocks_router

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "frontend" / "index.html"

app = FastAPI(
    title="AI Stock Predictor API",
    description="Predicts 5-day stock movement (Loss, Neutral, Profit) using technical indicators and XGBoost.",
    version="1.0.0"
)

# Enable CORS for frontend and cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(predict_router)
app.include_router(stocks_router)


@app.get("/", response_class=HTMLResponse)
def home():
    """
    Serve the interactive frontend dashboard at the root URL (DocMind deployment style).
    """
    if not INDEX_PATH.exists():
        return HTMLResponse("<h1>Frontend dashboard not found (frontend/index.html missing)</h1>", status_code=500)
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api")
def root():
    return {
        "name": "AI Stock Predictor API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
