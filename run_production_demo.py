#!/usr/bin/env python3
"""
Production Demo for Trader-X
Uses mock data when real data is unavailable to demonstrate full pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
from core.logger import logger
from core.ai_engine import ai_engine
from config.trading_config import TradingConfig
from data.market_data_enhanced import market_data_manager

def get_mock_fundamental_data(symbol):
    """Generate realistic mock fundamental data"""
    mock_data = {
        'TSLA': {
            'revenue_growth_yoy': 45.2,
            'revenue_growth_qoq': 28.1,
            'market_cap': 850000000000,
            'pe_ratio': 65.4,
            'revenue': 96773000000,
            'gross_margin': 0.21
        },
        'PLTR': {
            'revenue_growth_yoy': 38.7,
            'revenue_growth_qoq': 22.3,
            'market_cap': 45000000000,
            'pe_ratio': -1,  # Negative earnings
            'revenue': 2232000000,
            'gross_margin': 0.81
        },
        'NVDA': {
            'revenue_growth_yoy': 126.5,
            'revenue_growth_qoq': 88.4,
            'market_cap': 2800000000000,
            'pe_ratio': 78.2,
            'revenue': 79774000000,
            'gross_margin': 0.75
        }
    }
    return mock_data.get(symbol, {
        'revenue_growth_yoy': 15.0,
        'revenue_growth_qoq': 8.0,
        'market_cap': 50000000000,
        'pe_ratio': 25.0,
        'revenue': 10000000000,
        'gross_margin': 0.35
    })

def get_mock_technical_data(symbol):
    """Generate realistic mock technical data"""
    mock_data = {
        'TSLA': {
            'current_price': 248.50,
            'rsi': 58.3,
            'trend_strength': 'BULLISH',
            'volume_ratio': 1.45,
            'sma_20': 245.20,
            'sma_50': 238.10,
            'support_level': 235.00,
            'resistance_level': 265.00
        },
        'PLTR': {
            'current_price': 28.75,
            'rsi': 42.1,
            'trend_strength': 'NEUTRAL',
            'volume_ratio': 0.89,
            'sma_20': 29.10,
            'sma_50': 30.45,
            'support_level': 26.50,
            'resistance_level': 32.00
        },
        'NVDA': {
            'current_price': 1285.40,
            'rsi': 72.8,
            'trend_strength': 'STRONG_BULLISH',
            'volume_ratio': 2.15,
            'sma_20': 1245.30,
            'sma_50': 1180.75,
            'support_level': 1200.00,
            'resistance_level': 1350.00
        }
    }
    return mock_data.get(symbol, {
        'current_price': 100.00,
        'rsi': 50.0,
        'trend_strength': 'NEUTRAL',
        'volume_ratio': 1.0,
        'sma_20': 98.50,
        'sma_50': 97.00,
        'support_level': 95.00,
        'resistance_level': 105.00
    })

def run_production_demo():
    """Run the production demo with mock data fallback"""
    print("🚀 TRADER-X PRODUCTION DEMO")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now()}")
    print(f"📊 Mode: PAPER TRADING DEMO")
    print(f"🎯 Symbols: {TradingConfig.TEST_STOCKS[:3]}")
    print("=" * 60)
    
    try:
        # Check system status
        logger.info("Starting Trader-X production demo", "ORCHESTRATOR")
        
        # Check data connectivity
        connection_status = market_data_manager.get_connection_status()
        print(f"\n📊 DATA CONNECTIVITY:")
        print(f"   Primary Source: {connection_status['primary_data_source']}")
        print(f"   IB Gateway: {'✅ Connected' if connection_status['ib_gateway_connected'] else '❌ Disconnected'}")
