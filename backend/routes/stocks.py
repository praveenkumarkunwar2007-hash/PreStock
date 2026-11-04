from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from backend.services.stock_data import stock_data_service
from backend.services.news_service import news_service

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("", response_model=List[Dict[str, Any]])
def get_all_stocks():
    """
    List all tracked stocks across Indian and US exchanges.
    """
    return stock_data_service.list_stocks()


@router.get("/{symbol}/history")
def get_stock_history(symbol: str, limit: int = 30):
    """
    Get recent historical pricing and indicators for a symbol.
    """
    history = stock_data_service.get_stock_history(symbol, limit=limit)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Stock symbol '{symbol}' not found in datasets."
        )
    return {
        "symbol": symbol.upper(),
        "count": len(history),
        "history": history
    }


@router.get("/{symbol}/news")
def get_stock_news(symbol: str, limit: int = 6):
    """
    Fetch real-time news articles, publisher sources, and sentiment classification
    for a stock symbol from open-source financial feeds.
    """
    return news_service.fetch_stock_news(symbol, limit=limit)
