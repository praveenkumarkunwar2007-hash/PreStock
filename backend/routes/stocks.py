from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query

from backend.services.stock_data import stock_data_service
from backend.services.news_service import news_service
from backend.services.recommendation_service import recommendation_service

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("", response_model=List[Dict[str, Any]])
def get_all_stocks():
    """
    List all tracked stocks across Indian and US exchanges.
    """
    return stock_data_service.list_stocks()


@router.get("/search")
def search_stocks(q: str = Query(..., min_length=1, description="Company name, ticker, or IPO stock symbol")):
    """
    Search for any stock or IPO company across global exchanges (NSE, BSE, NASDAQ, NYSE).
    """
    return stock_data_service.search_stocks(q)


@router.get("/recommendations")
def get_recommendations(
    category: str = Query("all", description="Filter by asset type: all, stocks, ipos"),
    market: str = Query("all", description="Filter by exchange market: all, india, us"),
    limit: int = Query(10, ge=1, le=50, description="Max recommendations to return"),
    force_refresh: bool = Query(False, description="Bypass cache and force deep market scan")
):
    """
    Perform deep research across tracked stocks and top IPOs using XGBoost,
    multi-factor technical analysis, and sentiment catalysts to recommend
    the top profitable opportunities with transparent rationale.
    """
    cat_val = category.default if hasattr(category, "default") else category
    mkt_val = market.default if hasattr(market, "default") else market
    lim_val = limit.default if hasattr(limit, "default") else limit
    ref_val = force_refresh.default if hasattr(force_refresh, "default") else force_refresh

    return recommendation_service.scan_and_recommend(
        category=str(cat_val or "all"),
        market=str(mkt_val or "all"),
        limit=int(lim_val or 10),
        force_refresh=bool(ref_val or False)
    )


@router.get("/{symbol}/history")
def get_stock_history(symbol: str, limit: int = 30):
    """
    Get recent historical pricing and indicators for a symbol (tracked or live IPO).
    """
    history = stock_data_service.get_stock_history(symbol, limit=limit)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Stock symbol '{symbol}' not found in datasets or live market feeds."
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
