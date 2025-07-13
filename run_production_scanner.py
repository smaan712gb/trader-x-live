#!/usr/bin/env python3
"""
Production Market Scanner for Trader-X
Scans the entire market to find high-growth companies (35%+ revenue growth)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
from core.logger import logger
from core.orchestrator import orchestrator
from config.trading_config import TradingConfig
from data.market_data_enhanced import market_data_manager
import yfinance as yf
import pandas as pd

def get_market_universe():
    """Get a comprehensive list of stocks to scan"""
    # Start with major indices and expand
    universe = []
    
    # S&P 500 stocks
    try:
        sp500 = yf.Ticker("^GSPC")
        # For demo, we'll use a broader list of growth stocks and tech companies
        # In production, you'd fetch actual S&P 500 constituents
        
        # High-growth sectors and popular growth stocks
        growth_stocks = [
            # AI/Tech Growth
            'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'ANET', 'MRVL', 'QCOM',
            # Software/Cloud
            'CRM', 'NOW', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'PLTR',
            # EV/Clean Energy
            'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'BYD', 'CEG', 'ENPH',
            # Biotech/Healthcare
            'MRNA', 'BNTX', 'REGN', 'GILD', 'BIIB', 'VRTX', 'ILMN',
            # E-commerce/Digital
            'AMZN', 'SHOP', 'SQ', 'PYPL', 'ROKU', 'UBER', 'DASH',
            # Emerging Growth
            'RBLX', 'U', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM',
            # Infrastructure/5G
            'VRT', 'TOWER', 'AMT', 'CCI', 'SBAC',
            # Semiconductors
            'LRCX', 'KLAC', 'AMAT', 'MU', 'MCHP', 'ADI', 'TXN',
            # Cloud Infrastructure
            'NET', 'FSLY', 'ESTC', 'MDB', 'TEAM', 'ATLASSIAN',
            # Fintech
            'V', 'MA', 'ADYEN', 'FIS', 'FISV',
            # Gaming/Entertainment
            'NFLX', 'DIS', 'EA', 'ATVI', 'TTWO', 'RBLX',
            # Renewable Energy
            'FSLR', 'SEDG', 'RUN', 'PLUG', 'BE', 'ICLN'
        ]
        
        universe.extend(growth_stocks)
        
        # Add some mid-cap growth stocks
        midcap_growth = [
            'BILL', 'DOCN', 'GTLB', 'FROG', 'SMAR', 'PCTY', 'VEEV',
            'WDAY', 'ADSK', 'INTU', 'PANW', 'FTNT', 'CYBR', 'TENB'
        ]
        
        universe.extend(midcap_growth)
        
    except Exception as e:
        logger.warning(f"Could not fetch market indices: {e}", "SCANNER")
        # Fallback to a curated list
        universe = TradingConfig.TEST_STOCKS
    
    # Remove duplicates and return
    return list(set(universe))

def scan_for_growth_companies(universe, min_growth_yoy=35.0, min_growth_qoq=15.0, max_stocks=50):
    """Scan universe for high-growth companies"""
    print(f"\n🔍 SCANNING {len(universe)} STOCKS FOR HIGH-GROWTH COMPANIES")
    print("=" * 70)
    print(f"📊 Criteria: Revenue Growth YoY ≥ {min_growth_yoy}% OR QoQ ≥ {min_growth_qoq}%")
    print(f"🎯 Target: Find up to {max_stocks} qualifying companies")
    print("=" * 70)
    
    qualified_companies = []
    scanned_count = 0
    
    for i, symbol in enumerate(universe):
        if len(qualified_companies) >= max_stocks:
            print(f"\n✅ Reached target of {max_stocks} qualified companies")
            break
            
        try:
            scanned_count += 1
            print(f"\n📈 [{scanned_count}/{len(universe)}] Scanning {symbol}...")
            
            # Get fundamental data with rate limiting
            fundamental_data = market_data_manager.get_fundamental_data(symbol)
            
            if not fundamental_data or 'error' in fundamental_data:
                print(f"   ⚠️  No fundamental data available")
                time.sleep(1)  # Rate limiting
                continue
            
            # Extract growth metrics
            revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
            revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
            market_cap = fundamental_data.get('market_cap', 0)
            
            print(f"   Revenue Growth YoY: {revenue_growth_yoy:.1f}%")
            print(f"   Revenue Growth QoQ: {revenue_growth_qoq:.1f}%")
            print(f"   Market Cap: ${market_cap:,.0f}")
            
            # Check if company qualifies
            qualifies = False
            qualification_reason = ""
            
            if revenue_growth_yoy >= min_growth_yoy:
                qualifies = True
                qualification_reason = f"YoY Growth: {revenue_growth_yoy:.1f}%"
            elif revenue_growth_qoq >= min_growth_qoq:
                qualifies = True
                qualification_reason = f"QoQ Growth: {revenue_growth_qoq:.1f}%"
            
            if qualifies and market_cap > 1000000000:  # At least $1B market cap
                candidate = {
                    'symbol': symbol,
                    'revenue_growth_yoy': revenue_growth_yoy,
                    'revenue_growth_qoq': revenue_growth_qoq,
                    'market_cap': market_cap,
                    'qualification_reason': qualification_reason,
                    'fundamental_data': fundamental_data,
                    'discovery_rank': len(qualified_companies) + 1
                }
                qualified_companies.append(candidate)
                print(f"   ✅ QUALIFIED: {qualification_reason}")
                print(f"   🎯 Added to pipeline (#{len(qualified_companies)})")
            else:
                if market_cap <= 1000000000:
                    print(f"   ❌ Market cap too small: ${market_cap:,.0f}")
                else:
                    print(f"   ❌ Growth insufficient: YoY={revenue_growth_yoy:.1f}%, QoQ={revenue_growth_qoq:.1f}%")
            
            # Rate limiting to avoid overwhelming APIs
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error scanning {symbol}: {e}")
            logger.error(f"Error scanning {symbol}: {e}", "SCANNER")
            time.sleep(1)
            continue
    
    return qualified_companies

def run_production_scanner():
    """Run the production market scanner"""
    print("🚀 TRADER-X MARKET SCANNER")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now()}")
    print(f"📊 Mode: GROWTH COMPANY DISCOVERY")
    print(f"🎯 Target: 35%+ Revenue Growth Companies")
    print("=" * 60)
    
    try:
        # Initialize system
        logger.info("Starting Trader-X market scanner", "SCANNER")
        
        # Check data connectivity
        connection_status = market_data_manager.get_connection_status()
        print(f"\n📊 DATA CONNECTIVITY:")
        print(f"   Primary Source: {connection_status['primary_data_source']}")
        print(f"   IB Gateway: {'✅ Connected' if connection_status['ib_gateway_connected'] else '❌ Disconnected'}")
        print(f"   Fallback: {'✅ Available' if connection_status['fallback_available'] else '❌ Unavailable'}")
        
        # Get market universe
        print(f"\n🌐 BUILDING MARKET UNIVERSE...")
        universe = get_market_universe()
        print(f"   Universe Size: {len(universe)} stocks")
        print(f"   Sample: {universe[:10]}...")
        
        # Scan for high-growth companies
        qualified_companies = scan_for_growth_companies(
            universe, 
            min_growth_yoy=TradingConfig.MIN_REVENUE_GROWTH_YOY,
            min_growth_qoq=TradingConfig.MIN_REVENUE_GROWTH_QOQ,
            max_stocks=20  # Limit for demo
        )
        
        print(f"\n" + "=" * 70)
        print(f"🎯 GROWTH COMPANY DISCOVERY RESULTS")
        print("=" * 70)
        
        if not qualified_companies:
            print(f"❌ NO HIGH-GROWTH COMPANIES FOUND")
            print(f"   Criteria: YoY ≥ {TradingConfig.MIN_REVENUE_GROWTH_YOY}% OR QoQ ≥ {TradingConfig.MIN_REVENUE_GROWTH_QOQ}%")
            print(f"   Scanned: {len(universe)} companies")
            print(f"   Suggestion: Lower criteria or expand universe")
            return {'success': False, 'reason': 'No qualifying companies found'}
        
        print(f"✅ DISCOVERED {len(qualified_companies)} HIGH-GROWTH COMPANIES:")
        print()
        
        for i, company in enumerate(qualified_companies, 1):
            symbol = company['symbol']
            yoy = company['revenue_growth_yoy']
            qoq = company['revenue_growth_qoq']
            mcap = company['market_cap']
            reason = company['qualification_reason']
            
            print(f"   {i:2d}. {symbol:6s} | {reason:20s} | Market Cap: ${mcap:12,.0f}")
            print(f"       YoY: {yoy:6.1f}% | QoQ: {qoq:6.1f}%")
        
        # Now run the analysis pipeline on discovered companies
        print(f"\n🔬 RUNNING ANALYSIS PIPELINE ON DISCOVERED COMPANIES")
        print("=" * 70)
        
        # Phase 2: Technical Analysis
        logger.info("=== PHASE 2: TECHNICAL ANALYSIS ===", "SCANNER")
        print(f"\n📈 PHASE 2: TECHNICAL ANALYSIS")
        print("-" * 40)
        
        technical_qualified = []
        for company in qualified_companies[:10]:  # Analyze top 10
            symbol = company['symbol']
            try:
                print(f"\n🔧 Technical analysis for {symbol}...")
                
                # Get technical data
                technical_data = market_data_manager.get_technical_indicators(symbol)
                
                if technical_data and 'error' not in technical_data:
                    current_price = technical_data.get('current_price', 0)
                    rsi = technical_data.get('rsi', 50)
                    trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
                    volume_ratio = technical_data.get('volume_ratio', 1.0)
                    
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   RSI: {rsi:.1f}")
                    print(f"   Trend: {trend_strength}")
                    print(f"   Volume Ratio: {volume_ratio:.2f}x")
                    
                    # Technical scoring
                    technical_score = 50
                    if 30 <= rsi <= 70:
                        technical_score += 20
                    if trend_strength in ['BULLISH', 'STRONG_BULLISH']:
                        technical_score += 30
                    if volume_ratio > 1.2:
                        technical_score += 20
                    
                    company['technical_data'] = technical_data
                    company['technical_score'] = technical_score
                    
                    print(f"   Technical Score: {technical_score}/100")
                    
                    if technical_score >= 60:
                        technical_qualified.append(company)
                        print(f"   ✅ {symbol} qualified for AI analysis")
                    else:
                        print(f"   ⚠️  {symbol} has weak technical setup")
                else:
                    print(f"   ❌ {symbol} - No technical data available")
                    
            except Exception as e:
                print(f"   ❌ {symbol} - Technical analysis error: {e}")
                
            time.sleep(1)
        
        # Final Results
        print("\n" + "=" * 70)
        print("🎯 MARKET SCANNER RESULTS")
        print("=" * 70)
        
        print(f"✅ SCANNER COMPLETED SUCCESSFULLY")
        print(f"🌐 Universe Scanned: {len(universe)} stocks")
        print(f"🔍 Growth Companies Found: {len(qualified_companies)}")
        print(f"📈 Technical Qualified: {len(technical_qualified)}")
        
        if qualified_companies:
            print(f"\n🚀 TOP GROWTH DISCOVERIES:")
            for i, company in enumerate(qualified_companies[:5], 1):
                symbol = company['symbol']
                yoy = company['revenue_growth_yoy']
                reason = company['qualification_reason']
                print(f"   {i}. {symbol}: {reason}")
        
        if technical_qualified:
            print(f"\n📊 READY FOR AI ANALYSIS:")
            for company in technical_qualified:
                symbol = company['symbol']
                tech_score = company['technical_score']
                print(f"   • {symbol} (Technical Score: {tech_score}/100)")
        
        print(f"\n⏰ Scanner completed at: {datetime.now()}")
        print("=" * 70)
        
        return {
            'success': True,
            'universe_size': len(universe),
            'growth_companies_found': len(qualified_companies),
            'technical_qualified': len(technical_qualified),
            'discoveries': qualified_companies,
            'ready_for_ai': technical_qualified
        }
        
    except Exception as e:
        logger.error(f"Market scanner failed: {e}", "SCANNER")
        print(f"\n❌ SCANNER FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        result = run_production_scanner()
        if result.get('success'):
            print("\n🎉 MARKET SCANNER COMPLETED SUCCESSFULLY!")
            discoveries = result.get('growth_companies_found', 0)
            if discoveries > 0:
                print(f"🔍 Discovered {discoveries} high-growth companies meeting 35%+ criteria!")
            sys.exit(0)
        else:
            print(f"\n💥 MARKET SCANNER FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Scanner interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
