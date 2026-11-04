# AI Stock Predictor

An end-to-end Machine Learning pipeline and FastAPI application predicting 5-day stock movement (**Loss**, **Neutral**, **Profit**) across Indian (NSE/BSE) and US (NASDAQ/NYSE) markets using technical indicators and an optimized XGBoost classifier.

---

## 📁 Project Architecture

```
AI-Stock-Predictor/
├── backend/
│   ├── models/                # Trained model weights (.joblib) and metadata (.json)
│   ├── routes/                # FastAPI endpoint routers (predict, stocks)
│   ├── services/              # Inference engine (StockPredictor) & data provider
│   └── main.py                # FastAPI app entry point & CORS configuration
├── datasets/
│   ├── india/                 # indian_stocks.csv, indian_features.csv
│   └── usa/                   # us_stocks.csv, us_features.csv
├── frontend/
│   └── index.html             # Responsive dashboard with live chart & prediction meter
├── tests/
│   ├── test_preprocess.py     # Feature calculation & target generation tests
│   ├── test_predictor.py      # Inference & probability distribution tests
│   └── test_backend.py        # API endpoint & service integration tests
├── training/
│   ├── download_data.py       # Market data scraper via yfinance
│   ├── preprocess.py          # Technical indicator calculation & target labeling
│   ├── train_xgboost.py       # Chronological split & hyperparameter optimization
│   └── compare_random_forest.py # Random forest benchmark comparison
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Install Dependencies
```bash
# Activate existing virtualenv
.\venv\Scripts\Activate.ps1

# Or install dependencies in a fresh environment
pip install -r requirements.txt
```

### 2. Run Data Preprocessing (Feature Engineering)
```bash
python training/preprocess.py
```
Computes 15 technical indicators (RSI 14, Bollinger Bands, ATR, MACD, Moving Averages, Volatility, Volume Ratios) and defines future 5-day horizon labels:
- **Loss (`-1`)**: Future return $< -1\%$
- **Neutral (`0`)**: $-1\% \le \text{Return} \le +1\%$
- **Profit (`1`)**: Future return $> +1\%$

### 3. Model Training & Evaluation
```bash
# Train XGBoost v2 Classifier
python training/train_xgboost.py

# Benchmark Random Forest
python training/compare_random_forest.py
```

### 4. Run Unit & Integration Tests
```bash
python -m unittest discover -s tests
```

### 5. Launch the FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. View the Frontend Dashboard
Simply open [`frontend/index.html`](frontend/index.html) in your browser with the FastAPI server running.

---

## 🎯 API Endpoints

- `GET /health` - API health check status
- `GET /stocks` - List all tracked Indian and US stocks
- `GET /stocks/{symbol}/history?limit=30` - Recent price and indicator history
- `GET /predict/symbol/{symbol}` - 5-day forecast for a tracked symbol
- `POST /predict/custom` - 5-day forecast from custom OHLCV/indicator values
