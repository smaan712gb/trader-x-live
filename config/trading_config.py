"""
Trading Configuration Settings
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TradingConfig:
    # Phase 1: Signal Generation Criteria
    MIN_REVENUE_GROWTH_QOQ = 25.0  # Minimum quarterly revenue growth %
    MIN_REVENUE_GROWTH_YOY = 25.0  # Minimum yearly revenue growth %
    MIN_SENTIMENT_SCORE = 75.0     # Minimum positive sentiment score %
    
    # Financial Health Criteria
    MIN_CURRENT_RATIO = 1.2        # Minimum current ratio (current assets / current liabilities)
    MAX_DEBT_TO_EQUITY = 0.5       # Maximum debt-to-equity ratio
    MIN_CASH_TO_DEBT = 0.3         # Minimum cash-to-debt ratio for debt coverage
    MIN_PROFIT_MARGIN = 5.0        # Minimum profit margin %
    
    # Phase 2: Deep Analysis Parameters
    MIN_OPTIONS_OPEN_INTEREST = 1000  # Minimum open interest for key levels
    MIN_VOLUME_THRESHOLD = 1000000    # Minimum daily volume
    
    # Risk Management
    MAX_POSITION_SIZE = 0.05          # Maximum 5% of portfolio per position
    MAX_PORTFOLIO_RISK = 0.20         # Maximum 20% portfolio risk
    STOP_LOSS_PERCENTAGE = 0.08       # 8% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.15     # 15% take profit target
    
    # Timeframes for Technical Analysis
    TIMEFRAMES = ['1d', '4h', '1h', '15m']
    
    # Test Stocks
    TEST_STOCKS = ['TSLA', 'PLTR', 'NVDA', 'TSM', 'AVGO', 'CRWD', 'CEG', 'ANET', 'VRT', 'ASML']
    
    # Options Strategy (Theta Engine)
    MIN_THETA_PROBABILITY = 90.0      # Minimum 90% probability for theta trades
    MAX_THETA_POSITION_SIZE = 0.02    # Maximum 2% per theta trade
    
    # AI Engine Settings
    AI_CONFIDENCE_THRESHOLD = 0.75    # Minimum AI confidence for trade execution
    MAX_DAILY_TRADES = 5              # Maximum trades per day
    MIN_MARKET_SCORE = 30             # Minimum market score to proceed with trading
    
    # Additional Trading Parameters
    MARKET_HOURS_ONLY = False         # Allow after-hours trading
    ALLOW_POSITION_SCALING = False    # Don't add to existing positions
    MAX_POSITION_SIZE_PCT = 5.0       # Maximum 5% of portfolio per position
    MIN_ORDER_VALUE = 1000            # Minimum $1000 per order
    
    # Backtesting Parameters
    BACKTEST_START_DATE = '2023-01-01'
    BACKTEST_END_DATE = '2024-12-31'
    INITIAL_CAPITAL = 100000          # $100k starting capital
    
    # YouTube Sentiment Analysis
    MAX_VIDEOS_PER_STOCK = 50         # Maximum videos to analyze per stock
    VIDEO_AGE_DAYS = 30               # Only analyze videos from last 30 days
    MIN_VIDEO_VIEWS = 1000            # Minimum views for video consideration
    
    # ETF Tracking for Smart Money Flow
    TRACKED_ETFS = [
        'ARKK',  # ARK Innovation ETF
        'ARKQ',  # ARK Autonomous Technology & Robotics ETF
        'ARKW',  # ARK Next Generation Internet ETF
        'ARKG',  # ARK Genomics Revolution ETF
        'QQQ',   # Invesco QQQ Trust
        'XLK',   # Technology Select Sector SPDR Fund
        'SOXX',  # iShares Semiconductor ETF
    ]

class DatabaseConfig:
    # Supabase Table Names
    TRADES_TABLE = 'trades'
    MARKET_DATA_TABLE = 'market_data'
    SENTIMENT_DATA_TABLE = 'sentiment_data'
    PORTFOLIO_STATE_TABLE = 'portfolio_state'
    SYSTEM_LOGS_TABLE = 'system_logs'
    AI_MEMORY_TABLE = 'ai_memory'
    ETF_HOLDINGS_TABLE = 'etf_holdings'
    OPTIONS_DATA_TABLE = 'options_data'
    
    # Vector Embedding Configuration
    EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 embedding dimension
    SIMILARITY_THRESHOLD = 0.8  # Minimum similarity for memory retrieval
