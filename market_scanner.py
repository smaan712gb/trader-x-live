#!/usr/bin/env python3
"""
Professional Market Scanner - Like Finviz Pro
Combines market context, technical setups, and fundamental analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import time
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.logger import logger
from config.trading_config import TradingConfig

class MarketRegime(Enum):
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"

class TechnicalSetup(Enum):
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"
    MOMENTUM = "MOMENTUM"
    OVERSOLD_BOUNCE = "OVERSOLD_BOUNCE"
    CONSOLIDATION = "CONSOLIDATION"

@dataclass
class ScanResult:
    symbol: str
    setup_type: TechnicalSetup
    setup_strength: float  # 0-100
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    market_cap: float
    volume_ratio: float
    fundamental_score: float
    technical_score: float
    combined_score: float
    scan_timestamp: datetime

class MarketScanner:
    def __init__(self):
        self.market_context = {}
        self.scan_results = []
        self.last_scan_time = None
        
    def get_market_universe(self) -> List[str]:
        """Get comprehensive market universe for scanning"""
        # Major indices components + growth stocks
        universe = [
            # FAANG + Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'CSCO', 'AVGO', 'QCOM',
            
            # Growth Tech
            'PLTR', 'SNOW', 'CRWD', 'DDOG', 'NET', 'OKTA', 'ZS', 'PANW',
            'SHOP', 'SQ', 'PYPL', 'NOW', 'WDAY', 'VEEV', 'TEAM', 'ZM',
            
            # Fintech & Crypto
            'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'LC', 'MSTR', 'RIOT',
            
            # Healthcare & Biotech
            'JNJ', 'PFE', 'MRNA', 'BNTX', 'NVAX', 'TDOC', 'ILMN', 'REGN',
            'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'LLY', 'MRK',
            
            # Energy & Clean Tech
            'XOM', 'CVX', 'COP', 'EOG', 'ENPH', 'SEDG', 'FSLR', 'PLUG',
            'BE', 'CHPT', 'QS', 'RIVN', 'LCID', 'F', 'GM', 'TSLA',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
            'BRK-B', 'V', 'MA', 'AXP', 'COF', 'DFS', 'SYF',
            
            # Consumer & Retail
            'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX',
            'MCD', 'DIS', 'NFLX', 'ROKU', 'UBER', 'DASH', 'ABNB',
            
            # Industrial & Materials
            'CAT', 'DE', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX',
            'LMT', 'RTX', 'NOC', 'GD', 'FCX', 'NEM', 'GOLD',
            
            # Semiconductors
            'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT', 'LRCX',
            'KLAC', 'MRVL', 'XLNX', 'ADI', 'TXN', 'NXPI', 'MCHP',
            
            # ETFs for market context
            'SPY', 'QQQ', 'IWM', 'VTI', 'VIX', 'TLT', 'GLD', 'DXY'
        ]
        
        return list(set(universe))  # Remove duplicates
    
    def analyze_market_context(self) -> Dict[str, Any]:
        """Analyze overall market conditions - like Finviz market overview"""
        print("🌍 ANALYZING MARKET CONTEXT")
        print("=" * 40)
        
        try:
            context = {}
            
            # Major indices analysis
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'IWM': 'Russell 2000',
                'VIX': 'Volatility Index'
            }
            
            indices_data = {}
            for symbol, name in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="30d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        daily_change = ((current_price - prev_close) / prev_close) * 100
                        
                        # 20-day moving average
                        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                        trend = "ABOVE" if current_price > ma20 else "BELOW"
                        
                        indices_data[symbol] = {
                            'name': name,
                            'price': current_price,
                            'daily_change': daily_change,
                            'ma20': ma20,
                            'trend_vs_ma20': trend
                        }
                        
                        print(f"   {name:15s}: ${current_price:7.2f} ({daily_change:+5.2f}%) {trend} MA20")
                        
                except Exception as e:
                    print(f"   {name:15s}: Error - {e}")
            
            context['indices'] = indices_data
            
            # Market regime determination
            spy_data = indices_data.get('SPY', {})
            vix_data = indices_data.get('VIX', {})
            
            market_regime = self._determine_market_regime(spy_data, vix_data)
            context['market_regime'] = market_regime
            
            # Trading environment assessment
            trading_env = self._assess_trading_environment(indices_data)
            context['trading_environment'] = trading_env
            
            print(f"\n🎯 Market Regime: {market_regime}")
            print(f"📊 Trading Environment: {trading_env['recommendation']}")
            print(f"⚠️  Risk Level: {trading_env['risk_level']}")
            
            self.market_context = context
            return context
            
        except Exception as e:
            logger.error(f"Market context analysis failed: {e}", "SCANNER")
            return {'error': str(e)}
    
    def _determine_market_regime(self, spy_data: Dict, vix_data: Dict) -> MarketRegime:
        """Determine current market regime"""
        try:
            if not spy_data or not vix_data:
                return MarketRegime.SIDEWAYS
            
            spy_trend = spy_data.get('trend_vs_ma20', 'BELOW')
            spy_change = spy_data.get('daily_change', 0)
            vix_level = vix_data.get('price', 20)
            
            # High volatility regime
            if vix_level > 30:
                return MarketRegime.VOLATILE
            
            # Bull market conditions
            if spy_trend == 'ABOVE' and spy_change > 0 and vix_level < 20:
                return MarketRegime.BULL_MARKET
            
            # Bear market conditions
            if spy_trend == 'BELOW' and spy_change < -1 and vix_level > 25:
                return MarketRegime.BEAR_MARKET
            
            # Default to sideways
            return MarketRegime.SIDEWAYS
            
        except Exception:
            return MarketRegime.SIDEWAYS
    
    def _assess_trading_environment(self, indices_data: Dict) -> Dict[str, Any]:
        """Assess current trading environment"""
        try:
            spy_data = indices_data.get('SPY', {})
            vix_data = indices_data.get('VIX', {})
            
            spy_change = spy_data.get('daily_change', 0)
            vix_level = vix_data.get('price', 20)
            
            # Risk assessment
            if vix_level > 35:
                risk_level = "VERY_HIGH"
                recommendation = "DEFENSIVE"
            elif vix_level > 25:
                risk_level = "HIGH"
                recommendation = "CAUTIOUS"
            elif vix_level > 20:
                risk_level = "MODERATE"
                recommendation = "SELECTIVE"
            else:
                risk_level = "LOW"
                recommendation = "AGGRESSIVE"
            
            # Adjust based on market direction
            if spy_change < -2:
                recommendation = "DEFENSIVE"
                risk_level = "HIGH"
            elif spy_change > 2:
                if risk_level in ["LOW", "MODERATE"]:
                    recommendation = "AGGRESSIVE"
            
            return {
                'risk_level': risk_level,
                'recommendation': recommendation,
                'vix_level': vix_level,
                'market_direction': 'UP' if spy_change > 0 else 'DOWN'
            }
            
        except Exception:
            return {
                'risk_level': 'MODERATE',
                'recommendation': 'SELECTIVE',
                'vix_level': 20,
                'market_direction': 'NEUTRAL'
            }
    
    def scan_technical_setups(self, universe: List[str]) -> List[ScanResult]:
        """Scan for technical setups across the universe"""
        print(f"\n🔍 SCANNING TECHNICAL SETUPS")
        print("=" * 40)
        print(f"📊 Scanning {len(universe)} symbols...")
        
        scan_results = []
        processed = 0
        
        for symbol in universe:
            try:
                processed += 1
                if processed % 20 == 0:
                    print(f"   Progress: {processed}/{len(universe)} ({processed/len(universe)*100:.1f}%)")
                
                setup = self._analyze_technical_setup(symbol)
                if setup:
                    scan_results.append(setup)
                    
            except Exception as e:
                logger.debug(f"Technical analysis failed for {symbol}: {e}", "SCANNER")
                continue
        
        # Sort by combined score
        scan_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        print(f"✅ Technical scan complete: {len(scan_results)} setups found")
        return scan_results
    
    def _analyze_technical_setup(self, symbol: str) -> Optional[ScanResult]:
        """Analyze technical setup for a single symbol"""
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d")
            
            if len(hist) < 20:
                return None
            
            # Calculate technical indicators
            close = hist['Close']
            volume = hist['Volume']
            high = hist['High']
            low = hist['Low']
            
            # Moving averages
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean() if len(close) >= 50 else ma20
            
            # RSI
            rsi = self._calculate_rsi(close)
            
            # Volume analysis
            avg_volume = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Price levels
            current_price = close.iloc[-1]
            prev_close = close.iloc[-2]
            daily_change = ((current_price - prev_close) / prev_close) * 100
            
            # Support/Resistance
            recent_high = high.rolling(20).max().iloc[-1]
            recent_low = low.rolling(20).min().iloc[-1]
            
            # Identify setup type
            setup_type, setup_strength = self._identify_setup_type(
                current_price, ma20.iloc[-1], ma50.iloc[-1], rsi.iloc[-1],
                volume_ratio, daily_change, recent_high, recent_low
            )
            
            if setup_strength < 60:  # Minimum setup strength
                return None
            
            # Calculate entry, stop, target
            entry_price = current_price
            stop_loss = self._calculate_stop_loss(setup_type, current_price, recent_low, ma20.iloc[-1])
            target_price = self._calculate_target(setup_type, current_price, recent_high, stop_loss)
            
            risk_reward = (target_price - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
            
            # Get fundamental data
            fundamental_score = self._get_fundamental_score(symbol)
            
            # Technical score
            technical_score = self._calculate_technical_score(
                rsi.iloc[-1], volume_ratio, daily_change, setup_strength
            )
            
            # Combined score
            combined_score = (technical_score * 0.6) + (fundamental_score * 0.4)
            
            # Get market cap
            try:
                info = ticker.info
                market_cap = info.get('marketCap', 0)
            except:
                market_cap = 0
            
            return ScanResult(
                symbol=symbol,
                setup_type=setup_type,
                setup_strength=setup_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_reward=risk_reward,
                market_cap=market_cap,
                volume_ratio=volume_ratio,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                combined_score=combined_score,
                scan_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Technical setup analysis failed for {symbol}: {e}", "SCANNER")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _identify_setup_type(self, price: float, ma20: float, ma50: float, rsi: float,
                           volume_ratio: float, daily_change: float, 
                           recent_high: float, recent_low: float) -> tuple:
        """Identify technical setup type and strength"""
        
        setup_scores = {}
        
        # Breakout setup
        if price > recent_high * 0.99 and volume_ratio > 1.5 and daily_change > 2:
            setup_scores[TechnicalSetup.BREAKOUT] = 80 + min(20, volume_ratio * 5)
        
        # Pullback setup
        if (price > ma20 > ma50 and 
            recent_low < price < recent_high * 0.95 and 
            rsi > 40 and rsi < 60):
            setup_scores[TechnicalSetup.PULLBACK] = 70 + min(20, (60 - rsi))
        
        # Oversold bounce
        if rsi < 30 and daily_change > 1 and volume_ratio > 1.2:
            setup_scores[TechnicalSetup.OVERSOLD_BOUNCE] = 75 + min(15, (30 - rsi))
        
        # Momentum setup
        if (price > ma20 > ma50 and 
            daily_change > 3 and 
            volume_ratio > 2 and 
            rsi > 60):
            setup_scores[TechnicalSetup.MOMENTUM] = 85 + min(15, daily_change)
        
        # Reversal setup
        if ((rsi < 35 and daily_change > 2) or 
            (rsi > 65 and daily_change < -2)):
            setup_scores[TechnicalSetup.REVERSAL] = 65 + min(25, abs(daily_change) * 5)
        
        # Consolidation setup
        if (abs(daily_change) < 1 and 
            volume_ratio < 0.8 and 
            40 < rsi < 60):
            setup_scores[TechnicalSetup.CONSOLIDATION] = 50
        
        if not setup_scores:
            return TechnicalSetup.CONSOLIDATION, 30
        
        best_setup = max(setup_scores.items(), key=lambda x: x[1])
        return best_setup[0], min(100, best_setup[1])
    
    def _calculate_stop_loss(self, setup_type: TechnicalSetup, price: float, 
                           recent_low: float, ma20: float) -> float:
        """Calculate stop loss based on setup type"""
        if setup_type == TechnicalSetup.BREAKOUT:
            return max(recent_low, price * 0.92)  # 8% or recent low
        elif setup_type == TechnicalSetup.PULLBACK:
            return max(ma20 * 0.98, price * 0.95)  # MA20 or 5%
        elif setup_type == TechnicalSetup.OVERSOLD_BOUNCE:
            return recent_low * 0.98  # Just below recent low
        else:
            return price * 0.92  # Default 8% stop
    
    def _calculate_target(self, setup_type: TechnicalSetup, price: float, 
                        recent_high: float, stop_loss: float) -> float:
        """Calculate target price based on setup type"""
        risk = price - stop_loss
        
        if setup_type == TechnicalSetup.BREAKOUT:
            return price + (risk * 3)  # 3:1 risk/reward
        elif setup_type == TechnicalSetup.MOMENTUM:
            return price + (risk * 4)  # 4:1 risk/reward
        elif setup_type == TechnicalSetup.OVERSOLD_BOUNCE:
            return min(recent_high, price + (risk * 2))  # 2:1 or recent high
        else:
            return price + (risk * 2)  # Default 2:1 risk/reward
    
    def _get_fundamental_score(self, symbol: str) -> float:
        """Get fundamental score for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            score = 50  # Base score
            
            # Revenue growth
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth > 0.25:  # 25%+
                score += 30
            elif revenue_growth > 0.15:  # 15%+
                score += 20
            elif revenue_growth > 0.05:  # 5%+
                score += 10
            
            # Profit margins
            profit_margin = info.get('profitMargins', 0)
            if profit_margin > 0.2:  # 20%+
                score += 15
            elif profit_margin > 0.1:  # 10%+
                score += 10
            elif profit_margin < 0:  # Negative
                score -= 15
            
            # Debt levels
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity < 30:  # Low debt
                score += 10
            elif debt_to_equity > 100:  # High debt
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception:
            return 50  # Neutral score if data unavailable
    
    def _calculate_technical_score(self, rsi: float, volume_ratio: float, 
                                 daily_change: float, setup_strength: float) -> float:
        """Calculate technical score"""
        score = setup_strength * 0.4  # Base from setup strength
        
        # RSI contribution
        if 40 <= rsi <= 60:  # Neutral zone
            score += 15
        elif 30 <= rsi <= 70:  # Good zone
            score += 10
        elif rsi < 20 or rsi > 80:  # Extreme zone
            score += 5
        
        # Volume contribution
        if volume_ratio > 2:
            score += 20
        elif volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        
        # Momentum contribution
        if abs(daily_change) > 5:
            score += 15
        elif abs(daily_change) > 3:
            score += 10
        elif abs(daily_change) > 1:
            score += 5
        
        return max(0, min(100, score))
    
    def filter_by_market_context(self, scan_results: List[ScanResult]) -> List[ScanResult]:
        """Filter scan results based on market context"""
        if not self.market_context:
            return scan_results
        
        market_regime = self.market_context.get('market_regime', MarketRegime.SIDEWAYS)
        trading_env = self.market_context.get('trading_environment', {})
        risk_level = trading_env.get('risk_level', 'MODERATE')
        
        filtered_results = []
        
        for result in scan_results:
            # Market regime filters
            if market_regime == MarketRegime.BEAR_MARKET:
                # In bear market, prefer oversold bounces and reversals
                if result.setup_type in [TechnicalSetup.OVERSOLD_BOUNCE, TechnicalSetup.REVERSAL]:
                    filtered_results.append(result)
            
            elif market_regime == MarketRegime.BULL_MARKET:
                # In bull market, prefer breakouts and momentum
                if result.setup_type in [TechnicalSetup.BREAKOUT, TechnicalSetup.MOMENTUM, TechnicalSetup.PULLBACK]:
                    filtered_results.append(result)
            
            elif market_regime == MarketRegime.VOLATILE:
                # In volatile market, prefer high-quality setups only
                if result.combined_score > 80:
                    filtered_results.append(result)
            
            else:  # SIDEWAYS
                # In sideways market, all setups acceptable
                filtered_results.append(result)
            
            # Risk level filters
            if risk_level == "VERY_HIGH":
                # Only highest quality setups
                filtered_results = [r for r in filtered_results if r.combined_score > 85]
            elif risk_level == "HIGH":
                # High quality setups only
                filtered_results = [r for r in filtered_results if r.combined_score > 75]
        
        return filtered_results
    
    def run_full_scan(self) -> Dict[str, Any]:
        """Run complete market scan"""
        start_time = datetime.now()
        
        print("🚀 PROFESSIONAL MARKET SCANNER")
        print("=" * 50)
        print(f"🕐 Started: {start_time}")
        print("=" * 50)
        
        try:
            # Step 1: Market Context Analysis
            market_context = self.analyze_market_context()
            
            # Step 2: Get universe
            universe = self.get_market_universe()
            
            # Step 3: Technical setup scanning
            scan_results = self.scan_technical_setups(universe)
            
            # Step 4: Filter by market context
            filtered_results = self.filter_by_market_context(scan_results)
            
            # Step 5: Final ranking and display
            self._display_scan_results(filtered_results, market_context)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'duration': duration,
                'market_context': market_context,
                'total_scanned': len(universe),
                'setups_found': len(scan_results),
                'filtered_setups': len(filtered_results),
                'scan_results': filtered_results
            }
            
        except Exception as e:
            logger.error(f"Market scan failed: {e}", "SCANNER")
            return {'success': False, 'error': str(e)}
    
    def _display_scan_results(self, results: List[ScanResult], market_context: Dict):
        """Display scan results in a professional format"""
        print(f"\n📊 SCAN RESULTS")
        print("=" * 80)
        
        if not results:
            print("❌ No trading setups found matching current market conditions")
            return
        
        # Group by setup type
        setup_groups = {}
        for result in results:
            setup_type = result.setup_type.value
            if setup_type not in setup_groups:
                setup_groups[setup_type] = []
            setup_groups[setup_type].append(result)
        
        # Display by setup type
        for setup_type, group_results in setup_groups.items():
            print(f"\n🎯 {setup_type} SETUPS ({len(group_results)} found)")
            print("-" * 60)
            print(f"{'Symbol':<8} {'Score':<6} {'R/R':<6} {'Entry':<8} {'Stop':<8} {'Target':<8}")
            print("-" * 60)
            
            for result in group_results[:5]:  # Top 5 per setup type
                print(f"{result.symbol:<8} {result.combined_score:<6.1f} {result.risk_reward:<6.1f} "
                      f"${result.entry_price:<7.2f} ${result.stop_loss:<7.2f} ${result.target_price:<7.2f}")
        
        # Overall top picks
        top_picks = sorted(results, key=lambda x: x.combined_score, reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 TRADING OPPORTUNITIES")
        print("=" * 80)
        print(f"{'Rank':<4} {'Symbol':<8} {'Setup':<12} {'Score':<6} {'R/R':<6} {'Entry':<8} {'Market Cap':<10}")
        print("-" * 80)
        
        for i, result in enumerate(top_picks, 1):
            market_cap_str = f"${result.market_cap/1e9:.1f}B" if result.market_cap > 0 else "N/A"
            print(f"{i:<4} {result.symbol:<8} {result.setup_type.value:<12} {result.combined_score:<6.1f} "
                  f"{result.risk_reward:<6.1f} ${result.entry_price:<7.2f} {market_cap_str:<10}")
        
        # Market context summary
        regime = market_context.get('market_regime', 'UNKNOWN')
        trading_env = market_context.get('trading_environment', {})
        recommendation = trading_env.get('recommendation', 'SELECTIVE')
        
        print(f"\n📈 MARKET CONTEXT SUMMARY")
        print("-" * 40)
        print(f"Market Regime: {regime}")
        print(f"Trading Recommendation: {recommendation}")
        print(f"Risk Level: {trading_env.get('risk_level', 'MODERATE')}")
        print(f"Setups Found: {len(results)}")
        print(f"High-Quality Setups: {len([r for r in results if r.combined_score > 80])}")

def main():
    """Main scanner execution"""
    scanner = MarketScanner()
    result = scanner.run_full_scan()
    
    if result.get('success'):
        print(f"\n✅ Scan completed successfully in {result['duration']:.1f} seconds")
        print(f"📊 {result['filtered_setups']} trading opportunities identified")
    else:
        print(f"\n❌ Scan failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
