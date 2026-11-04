from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from backend.services.stock_data import stock_data_service

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
