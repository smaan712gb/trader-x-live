#!/usr/bin/env python3
"""
Production-Ready Market Scanner for Trader-X
Robust growth company discovery with proper error handling
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
from core.logger import logger
from config.trading_config import TradingConfig
from data.market_data_production import production_market_data
import yfinance as yf

def get_production_universe():
    """Get production-ready market universe"""
    # Start with growth-focused sectors
    universe = []
    
    # High-growth sectors
    growth_stocks = [
        # AI/Semiconductor Leaders
        'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'ANET', 'MRVL', 'QCOM', 'LRCX', 'KLAC',
        # Software/Cloud Growth
        'CRM', 'NOW', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'PLTR', 'NET', 'FSLY',
        # Clean Energy/EV
        'TSLA', 'ENPH', 'FSLR', 'SEDG', 'CEG', 'RUN', 'PLUG', 'BE',
        # Biotech/Healthcare Innovation
        'MRNA', 'BNTX', 'REGN', 'GILD', 'VRTX', 'ILMN', 'BIIB',
        # Digital/E-commerce
        'SHOP', 'SQ', 'PYPL', 'ROKU', 'UBER', 'DASH', 'COIN',
        # Emerging Growth
        'RBLX', 'U', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'BILL',
        # Infrastructure/5G
        'VRT', 'TOWER', 'AMT', 'CCI', 'SBAC',
        # Fintech Innovation
        'V', 'MA', 'FIS', 'FISV', 'ADYEN'
    ]
    
    universe.extend(growth_stocks)
    
    # Remove duplicates
    universe = list(set(universe))
    
    logger.info(f"Built production universe with {len(universe)} stocks", "SCANNER")
    return universe

def scan_for_growth_companies(universe, min_growth_yoy=35.0, min_growth_qoq=35.0, max_companies=15):
    """Production scanner for high-growth companies"""
    print(f"\n🔍 PRODUCTION GROWTH SCANNER")
    print("=" * 70)
    print(f"📊 Universe: {len(universe)} stocks")
    print(f"🎯 Criteria: Revenue Growth YoY ≥ {min_growth_yoy}% OR QoQ ≥ {min_growth_qoq}%")
    print(f"🏆 Target: Up to {max_companies} qualifying companies")
    print("=" * 70)
    
    qualified_companies = []
    processed_count = 0
    error_count = 0
    
    # Get connection status
    status = production_market_data.get_connection_status()
    print(f"\n📊 DATA CONNECTIVITY:")
    print(f"   Primary Source: {status['primary_data_source']}")
    print(f"   IB Gateway: {'✅ Connected' if status['ib_gateway_connected'] else '❌ Disconnected'}")
    print(f"   Rate Limits: {status['rate_limits']['yahoo_requests_today']}/{status['rate_limits']['daily_limit']}")
    
    for i, symbol in enumerate(universe):
        if len(qualified_companies) >= max_companies:
            print(f"\n✅ Reached target of {max_companies} qualified companies")
            break
            
        if error_count > 20:  # Stop if too many consecutive errors
            print(f"\n⚠️  Too many errors ({error_count}), stopping scan")
            break
        
        try:
            processed_count += 1
            print(f"\n📈 [{processed_count}/{len(universe)}] Analyzing {symbol}...")
            
            # Get fundamental data
            fundamental_data = production_market_data.get_fundamental_data(symbol)
            
            if not fundamental_data or 'error' in fundamental_data:
                error_count += 1
                print(f"   ❌ No fundamental data: {fundamental_data.get('error', 'Unknown error')}")
                continue
            
            # Reset error count on success
            error_count = 0
            
            # Extract key metrics
            revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
            revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
            market_cap = fundamental_data.get('market_cap', 0)
            pe_ratio = fundamental_data.get('pe_ratio', 0)
            source = fundamental_data.get('source', 'unknown')
            
            print(f"   📊 Data Source: {source.upper()}")
            print(f"   💰 Market Cap: ${market_cap:,.0f}")
            print(f"   📈 Revenue Growth YoY: {revenue_growth_yoy:.1f}%")
            print(f"   📊 Revenue Growth QoQ: {revenue_growth_qoq:.1f}%")
            print(f"   💹 P/E Ratio: {pe_ratio:.1f}")
            
            # Check qualification criteria
            qualifies = False
            qualification_reason = ""
            
            # Primary criteria: High revenue growth
            if revenue_growth_yoy >= min_growth_yoy:
                qualifies = True
                qualification_reason = f"YoY Growth: {revenue_growth_yoy:.1f}%"
            elif revenue_growth_qoq >= min_growth_qoq:
                qualifies = True
                qualification_reason = f"QoQ Growth: {revenue_growth_qoq:.1f}%"
            
            # Additional filters
            min_market_cap = 1_000_000_000  # $1B minimum
            max_pe_ratio = 200  # Reasonable P/E limit
            
            if qualifies:
                if market_cap < min_market_cap:
                    print(f"   ❌ Market cap too small: ${market_cap:,.0f} < ${min_market_cap:,.0f}")
                    qualifies = False
                elif pe_ratio > max_pe_ratio and pe_ratio > 0:
                    print(f"   ❌ P/E ratio too high: {pe_ratio:.1f} > {max_pe_ratio}")
                    qualifies = False
            
            if qualifies:
                # Calculate growth score
                growth_score = min(100, (revenue_growth_yoy + revenue_growth_qoq) / 2)
                
                candidate = {
                    'symbol': symbol,
                    'revenue_growth_yoy': revenue_growth_yoy,
                    'revenue_growth_qoq': revenue_growth_qoq,
                    'market_cap': market_cap,
                    'pe_ratio': pe_ratio,
                    'growth_score': growth_score,
                    'qualification_reason': qualification_reason,
                    'data_source': source,
                    'fundamental_data': fundamental_data,
                    'discovery_rank': len(qualified_companies) + 1,
                    'discovery_time': datetime.now().isoformat()
                }
                
                qualified_companies.append(candidate)
                print(f"   ✅ QUALIFIED: {qualification_reason}")
                print(f"   🎯 Growth Score: {growth_score:.1f}/100")
                print(f"   🏆 Added to pipeline (#{len(qualified_companies)})")
                
                # Log discovery
                logger.info(f"Growth company discovered: {symbol} - {qualification_reason}", "SCANNER", {
                    'symbol': symbol,
                    'growth_score': growth_score,
                    'market_cap': market_cap,
                    'data_source': source
                })
            else:
                print(f"   ❌ Does not meet criteria: YoY={revenue_growth_yoy:.1f}%, QoQ={revenue_growth_qoq:.1f}%")
            
        except Exception as e:
            error_count += 1
            print(f"   💥 Error analyzing {symbol}: {e}")
            logger.error(f"Scanner error for {symbol}: {e}", "SCANNER")
            continue
    
    return qualified_companies, processed_count, error_count

def run_technical_analysis(qualified_companies):
    """Run technical analysis on qualified companies"""
    print(f"\n🔧 TECHNICAL ANALYSIS PHASE")
    print("=" * 50)
    
    technical_qualified = []
    
    for i, company in enumerate(qualified_companies[:10], 1):  # Analyze top 10
        symbol = company['symbol']
        try:
            print(f"\n📊 [{i}/10] Technical analysis: {symbol}")
            
            technical_data = production_market_data.get_technical_indicators(symbol)
            
            if technical_data and 'error' not in technical_data:
                current_price = technical_data.get('current_price', 0)
                rsi = technical_data.get('rsi', 50)
                trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
                volume_ratio = technical_data.get('volume_ratio', 1.0)
                
                print(f"   💰 Price: ${current_price:.2f}")
                print(f"   📈 RSI: {rsi:.1f}")
                print(f"   🎯 Trend: {trend_strength}")
                print(f"   📊 Volume: {volume_ratio:.2f}x avg")
                
                # Technical scoring
                technical_score = 0
                
                # RSI scoring (prefer not overbought/oversold)
                if 30 <= rsi <= 70:
                    technical_score += 25
                elif 25 <= rsi <= 75:
                    technical_score += 15
                
                # Trend scoring
                if trend_strength == 'BULLISH':
                    technical_score += 35
                elif trend_strength == 'NEUTRAL':
                    technical_score += 20
                
                # Volume scoring
                if volume_ratio > 1.5:
                    technical_score += 25
                elif volume_ratio > 1.2:
                    technical_score += 15
                elif volume_ratio > 0.8:
                    technical_score += 10
                
                # Momentum bonus
                if trend_strength == 'BULLISH' and rsi > 50 and volume_ratio > 1.2:
                    technical_score += 15
                
                company['technical_data'] = technical_data
                company['technical_score'] = technical_score
                
                print(f"   🎯 Technical Score: {technical_score}/100")
                
                if technical_score >= 50:  # Minimum technical threshold
                    technical_qualified.append(company)
                    print(f"   ✅ Technical qualification passed")
                else:
                    print(f"   ⚠️  Weak technical setup")
                    
            else:
                print(f"   ❌ No technical data available")
                
        except Exception as e:
            print(f"   💥 Technical analysis error: {e}")
            logger.error(f"Technical analysis error for {symbol}: {e}", "SCANNER")
    
    return technical_qualified

def run_production_scanner():
    """Main production scanner function"""
    start_time = datetime.now()
    
    print("🚀 TRADER-X PRODUCTION GROWTH SCANNER")
    print("=" * 60)
    print(f"🕐 Started: {start_time}")
    print(f"📊 Mode: HIGH-GROWTH COMPANY DISCOVERY")
    print(f"🎯 Objective: Find 35%+ revenue growth companies")
    print("=" * 60)
    
    try:
        logger.info("Starting production growth scanner", "SCANNER")
        
        # Build universe
        universe = get_production_universe()
        
        # Phase 1: Growth Discovery
        print(f"\n🔍 PHASE 1: GROWTH DISCOVERY")
        print("-" * 40)
        
        qualified_companies, processed, errors = scan_for_growth_companies(
            universe,
            min_growth_yoy=TradingConfig.MIN_REVENUE_GROWTH_YOY,
            min_growth_qoq=TradingConfig.MIN_REVENUE_GROWTH_QOQ,
            max_companies=15
        )
        
        # Phase 2: Technical Analysis
        technical_qualified = []
        if qualified_companies:
            technical_qualified = run_technical_analysis(qualified_companies)
        
        # Results Summary
        print(f"\n" + "=" * 70)
        print(f"🎯 PRODUCTION SCANNER RESULTS")
        print("=" * 70)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  EXECUTION SUMMARY:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Universe Size: {len(universe)} stocks")
        print(f"   Processed: {processed} stocks")
        print(f"   Errors: {errors}")
        print(f"   Success Rate: {((processed - errors) / processed * 100):.1f}%")
        
        print(f"\n📊 DISCOVERY RESULTS:")
        print(f"   Growth Companies Found: {len(qualified_companies)}")
        print(f"   Technical Qualified: {len(technical_qualified)}")
        
        if qualified_companies:
            print(f"\n🚀 TOP GROWTH DISCOVERIES:")
            for i, company in enumerate(qualified_companies[:5], 1):
                symbol = company['symbol']
                reason = company['qualification_reason']
                score = company['growth_score']
                mcap = company['market_cap']
                source = company['data_source']
                
                print(f"   {i}. {symbol:6s} | {reason:20s} | Score: {score:5.1f} | ${mcap:12,.0f} | {source}")
        
        if technical_qualified:
            print(f"\n📈 READY FOR AI ANALYSIS:")
            for company in technical_qualified:
                symbol = company['symbol']
                growth_score = company['growth_score']
                tech_score = company['technical_score']
                print(f"   • {symbol}: Growth={growth_score:.1f}, Technical={tech_score}/100")
        
        print(f"\n✅ Scanner completed successfully at {end_time}")
        print("=" * 70)
        
        # Log final results
        logger.info(f"Production scanner completed: {len(qualified_companies)} growth companies found", "SCANNER", {
            'duration_seconds': duration,
            'processed_count': processed,
            'error_count': errors,
            'growth_companies': len(qualified_companies),
            'technical_qualified': len(technical_qualified)
        })
        
        return {
            'success': True,
            'duration': duration,
            'universe_size': len(universe),
            'processed': processed,
            'errors': errors,
            'growth_companies': qualified_companies,
            'technical_qualified': technical_qualified,
            'summary': {
                'total_discovered': len(qualified_companies),
                'ready_for_ai': len(technical_qualified),
                'success_rate': (processed - errors) / processed * 100 if processed > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Production scanner failed: {e}", "SCANNER")
        print(f"\n💥 SCANNER FAILED: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        # Cleanup
        try:
            production_market_data.cleanup()
        except:
            pass

if __name__ == "__main__":
    try:
        result = run_production_scanner()
        
        if result.get('success'):
            discoveries = result.get('summary', {}).get('total_discovered', 0)
            ready_for_ai = result.get('summary', {}).get('ready_for_ai', 0)
            
            print(f"\n🎉 SUCCESS: Found {discoveries} high-growth companies!")
            if ready_for_ai > 0:
                print(f"🤖 {ready_for_ai} companies ready for AI analysis")
            
            sys.exit(0)
        else:
            print(f"\n💥 FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Scanner interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.error(f"Unexpected scanner error: {e}", "SCANNER")
        sys.exit(1)
