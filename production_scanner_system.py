#!/usr/bin/env python3
"""
Production Market Scanner System
Integrates with IB Gateway for real-time data and avoids rate limits
Identifies next big winners like PLTR, CRWD, NVDA
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

from core.logger import logger
from data.ib_gateway import IBGatewayConnector
from config.trading_config import TradingConfig

class MarketSector(Enum):
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare" 
    FINANCIAL = "Financial"
    ENERGY = "Energy"
    CONSUMER = "Consumer"
    INDUSTRIAL = "Industrial"

class SetupType(Enum):
    BREAKOUT = "Breakout"
    PULLBACK = "Pullback"
    MOMENTUM = "Momentum"
    REVERSAL = "Reversal"
    ACCUMULATION = "Accumulation"

@dataclass
class ScannerResult:
    symbol: str
    company_name: str
    sector: MarketSector
    setup_type: SetupType
    current_price: float
    market_cap: float
    volume_ratio: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    rsi: float
    macd_signal: str
    support_level: float
    resistance_level: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    scanner_score: float
    growth_potential: str
    risk_level: str
    catalyst_events: List[str]
    scan_timestamp: datetime

class ProductionScanner:
    def __init__(self):
        self.ib_gateway = IBGatewayConnector(port=4002)  # Use correct IB Gateway port
        self.config = TradingConfig()
        self.scan_results = []
        
        # Define our universe of potential winners
        self.growth_universe = {
            MarketSector.TECHNOLOGY: [
                # AI & Cloud
                'PLTR', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'PANW',
                'MDB', 'ESTC', 'FSLY', 'TWLO', 'DOCN', 'TEAM', 'ZM',
                
                # Semiconductors
                'NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL', 'AMAT', 'LRCX', 'KLAC',
                
                # Software
                'CRM', 'NOW', 'VEEV', 'WDAY', 'ADBE', 'ORCL', 'MSFT'
            ],
            
            MarketSector.HEALTHCARE: [
                # Biotech
                'MRNA', 'BNTX', 'NVAX', 'CRSP', 'EDIT', 'NTLA', 'BEAM',
                'ILMN', 'PACB', 'TDOC', 'VEEV', 'DXCM', 'ISRG'
            ],
            
            MarketSector.FINANCIAL: [
                # Fintech
                'SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'LC',
                'BILL', 'PAYO', 'NU', 'MELI', 'SE'
            ],
            
            MarketSector.ENERGY: [
                # Clean Energy
                'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'PLUG', 'BE', 'BLDP',
                'FCEL', 'CHPT', 'BLNK', 'QS', 'STEM'
            ],
            
            MarketSector.CONSUMER: [
                # E-commerce & Consumer Tech
                'SHOP', 'ROKU', 'UBER', 'DASH', 'ABNB', 'PTON', 'NFLX'
            ],
            
            MarketSector.INDUSTRIAL: [
                # Automation & Robotics
                'IRBT', 'ZBRA', 'TER', 'OMCL'
            ]
        }
    
    async def initialize_connection(self) -> bool:
        """Initialize IB Gateway connection"""
        try:
            print("🔌 Connecting to IB Gateway...")
            success = self.ib_gateway.connect()  # Remove await - it's not async
            if success:
                print("✅ Connected to IB Gateway successfully")
                return True
            else:
                print("❌ Failed to connect to IB Gateway")
                return False
        except Exception as e:
            logger.error(f"IB Gateway connection failed: {e}", "SCANNER")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time market data for symbol"""
        try:
            # Get current market data
            quote = self.ib_gateway.get_market_data(symbol)
            if not quote:
                return None
            
            # Get historical data for technical analysis
            hist_data = self.ib_gateway.get_historical_data(
                symbol, 
                duration="30 D",
                bar_size="1 day"
            )
            
            if hist_data is None or len(hist_data) < 20:
                return None
            
            return {
                'quote': quote,
                'historical': hist_data,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.debug(f"Market data fetch failed for {symbol}: {e}", "SCANNER")
            return None
    
    def calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            close = hist_data['close']
            high = hist_data['high']
            low = hist_data['low']
            volume = hist_data['volume']
            
            # RSI
            rsi = self._calculate_rsi(close)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
            
            # Support/Resistance
            support = low.rolling(20).min().iloc[-1]
            resistance = high.rolling(20).max().iloc[-1]
            
            # Volume analysis
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price changes
            current_price = close.iloc[-1]
            price_1d = ((current_price / close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
            price_5d = ((current_price / close.iloc[-6]) - 1) * 100 if len(close) > 5 else 0
            price_20d = ((current_price / close.iloc[-21]) - 1) * 100 if len(close) > 20 else 0
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd_line': macd_line.iloc[-1] if not macd_line.empty else 0,
                'macd_signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0,
                'support': support,
                'resistance': resistance,
                'volume_ratio': volume_ratio,
                'price_1d': price_1d,
                'price_5d': price_5d,
                'price_20d': price_20d,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.debug(f"Technical indicator calculation failed: {e}", "SCANNER")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def identify_setup_type(self, indicators: Dict[str, float]) -> Tuple[SetupType, float]:
        """Identify technical setup type and strength"""
        rsi = indicators.get('rsi', 50)
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        volume_ratio = indicators.get('volume_ratio', 1)
        price_1d = indicators.get('price_1d', 0)
        price_5d = indicators.get('price_5d', 0)
        current_price = indicators.get('current_price', 0)
        resistance = indicators.get('resistance', 0)
        support = indicators.get('support', 0)
        
        setup_scores = {}
        
        # Breakout setup
        if (current_price > resistance * 0.99 and 
            volume_ratio > 1.5 and 
            price_1d > 2):
            setup_scores[SetupType.BREAKOUT] = 80 + min(20, volume_ratio * 5)
        
        # Momentum setup
        if (macd_line > macd_signal and 
            rsi > 60 and rsi < 80 and 
            price_5d > 5 and 
            volume_ratio > 1.2):
            setup_scores[SetupType.MOMENTUM] = 75 + min(20, price_5d)
        
        # Pullback setup
        if (price_5d > 0 and 
            price_1d < 0 and 
            rsi > 40 and rsi < 60 and 
            current_price > support * 1.02):
            setup_scores[SetupType.PULLBACK] = 70 + min(20, abs(price_1d))
        
        # Reversal setup
        if ((rsi < 30 and price_1d > 1) or 
            (rsi > 70 and price_1d < -1)):
            setup_scores[SetupType.REVERSAL] = 65 + min(25, abs(price_1d) * 3)
        
        # Accumulation setup
        if (volume_ratio > 1.3 and 
            abs(price_1d) < 2 and 
            rsi > 45 and rsi < 55):
            setup_scores[SetupType.ACCUMULATION] = 60 + min(15, volume_ratio * 5)
        
        if not setup_scores:
            return SetupType.ACCUMULATION, 30
        
        best_setup = max(setup_scores.items(), key=lambda x: x[1])
        return best_setup[0], min(100, best_setup[1])
    
    def calculate_entry_exit_levels(self, setup_type: SetupType, indicators: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and target levels"""
        current_price = indicators.get('current_price', 0)
        support = indicators.get('support', current_price * 0.95)
        resistance = indicators.get('resistance', current_price * 1.05)
        
        if setup_type == SetupType.BREAKOUT:
            entry = current_price
            stop_loss = max(support, current_price * 0.92)
            target = current_price + ((current_price - stop_loss) * 3)  # 3:1 R/R
            
        elif setup_type == SetupType.MOMENTUM:
            entry = current_price
            stop_loss = max(support, current_price * 0.94)
            target = current_price + ((current_price - stop_loss) * 2.5)  # 2.5:1 R/R
            
        elif setup_type == SetupType.PULLBACK:
            entry = current_price
            stop_loss = support * 0.98
            target = min(resistance, current_price + ((current_price - stop_loss) * 2))
            
        else:  # REVERSAL, ACCUMULATION
            entry = current_price
            stop_loss = current_price * 0.92
            target = current_price + ((current_price - stop_loss) * 2)
        
        return entry, stop_loss, target
    
    def calculate_scanner_score(self, setup_strength: float, indicators: Dict[str, float], 
                              market_cap: float) -> float:
        """Calculate overall scanner score"""
        base_score = setup_strength * 0.4
        
        # Technical momentum score
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1)
        price_5d = indicators.get('price_5d', 0)
        
        momentum_score = 0
        if 50 <= rsi <= 70:  # Good momentum zone
            momentum_score += 20
        elif 30 <= rsi <= 80:  # Acceptable zone
            momentum_score += 10
        
        if volume_ratio > 2:
            momentum_score += 20
        elif volume_ratio > 1.5:
            momentum_score += 15
        elif volume_ratio > 1.2:
            momentum_score += 10
        
        if price_5d > 10:
            momentum_score += 15
        elif price_5d > 5:
            momentum_score += 10
        elif price_5d > 0:
            momentum_score += 5
        
        # Market cap score (sweet spot for growth)
        cap_score = 0
        if 1e9 <= market_cap <= 50e9:  # $1B - $50B sweet spot
            cap_score = 20
        elif 500e6 <= market_cap <= 100e9:  # $500M - $100B acceptable
            cap_score = 15
        elif market_cap <= 200e9:  # Up to $200B
            cap_score = 10
        
        total_score = base_score + (momentum_score * 0.4) + (cap_score * 0.2)
        return min(100, max(0, total_score))
    
    def assess_growth_potential(self, scanner_score: float, setup_type: SetupType, 
                              sector: MarketSector) -> str:
        """Assess growth potential"""
        # High-growth sectors
        growth_sectors = [MarketSector.TECHNOLOGY, MarketSector.HEALTHCARE]
        
        if scanner_score > 80 and sector in growth_sectors:
            return "EXPLOSIVE"
        elif scanner_score > 70:
            return "HIGH"
        elif scanner_score > 60:
            return "MODERATE"
        else:
            return "LOW"
    
    def get_catalyst_events(self, symbol: str, sector: MarketSector) -> List[str]:
        """Identify potential catalyst events"""
        catalysts = []
        
        # Sector-specific catalysts
        if sector == MarketSector.TECHNOLOGY:
            catalysts.extend(["AI adoption", "Cloud migration", "Digital transformation"])
        elif sector == MarketSector.HEALTHCARE:
            catalysts.extend(["Drug approvals", "Clinical trials", "Healthcare digitization"])
        elif sector == MarketSector.FINANCIAL:
            catalysts.extend(["Fintech adoption", "Digital payments", "Crypto integration"])
        elif sector == MarketSector.ENERGY:
            catalysts.extend(["Green transition", "EV adoption", "Energy storage"])
        
        # Add earnings catalyst (next earnings typically within 90 days)
        catalysts.append("Upcoming earnings")
        
        return catalysts[:3]  # Top 3 catalysts
    
    async def scan_symbol(self, symbol: str, sector: MarketSector) -> Optional[ScannerResult]:
        """Scan individual symbol for trading opportunities"""
        try:
            # Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return None
            
            quote = market_data['quote']
            hist_data = market_data['historical']
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(hist_data)
            if not indicators:
                return None
            
            # Identify setup
            setup_type, setup_strength = self.identify_setup_type(indicators)
            
            # Calculate entry/exit levels
            entry, stop_loss, target = self.calculate_entry_exit_levels(setup_type, indicators)
            
            # Calculate risk/reward
            risk_reward = (target - entry) / (entry - stop_loss) if entry > stop_loss else 0
            
            # Get market cap (estimate from price and shares outstanding)
            current_price = indicators['current_price']
            market_cap = quote.get('marketCap', current_price * 1e9)  # Fallback estimate
            
            # Calculate scanner score
            scanner_score = self.calculate_scanner_score(setup_strength, indicators, market_cap)
            
            # Assess growth potential and risk
            growth_potential = self.assess_growth_potential(scanner_score, setup_type, sector)
            risk_level = "HIGH" if market_cap < 2e9 else "MODERATE" if market_cap < 10e9 else "LOW"
            
            # Get catalysts
            catalysts = self.get_catalyst_events(symbol, sector)
            
            # MACD signal
            macd_signal = "BULLISH" if indicators.get('macd_line', 0) > indicators.get('macd_signal', 0) else "BEARISH"
            
            return ScannerResult(
                symbol=symbol,
                company_name=quote.get('companyName', symbol),
                sector=sector,
                setup_type=setup_type,
                current_price=current_price,
                market_cap=market_cap,
                volume_ratio=indicators['volume_ratio'],
                price_change_1d=indicators['price_1d'],
                price_change_5d=indicators['price_5d'],
                price_change_20d=indicators['price_20d'],
                rsi=indicators['rsi'],
                macd_signal=macd_signal,
                support_level=indicators['support'],
                resistance_level=indicators['resistance'],
                entry_price=entry,
                stop_loss=stop_loss,
                target_price=target,
                risk_reward_ratio=risk_reward,
                scanner_score=scanner_score,
                growth_potential=growth_potential,
                risk_level=risk_level,
                catalyst_events=catalysts,
                scan_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Symbol scan failed for {symbol}: {e}", "SCANNER")
            return None
    
    async def run_full_scan(self, min_score: float = 60) -> List[ScannerResult]:
        """Run complete market scan"""
        print("🚀 PRODUCTION MARKET SCANNER")
        print("=" * 50)
        print("🎯 Finding next PLTR, CRWD, NVDA")
        print(f"📊 Minimum Score: {min_score}")
        print("=" * 50)
        
        # Initialize IB connection
        if not await self.initialize_connection():
            print("❌ Cannot proceed without IB Gateway connection")
            return []
        
        all_results = []
        
        for sector, symbols in self.growth_universe.items():
            print(f"\n🔍 Scanning {sector.value} ({len(symbols)} symbols)...")
            sector_results = []
            
            for symbol in symbols:
                try:
                    result = await self.scan_symbol(symbol, sector)
                    if result and result.scanner_score >= min_score:
                        sector_results.append(result)
                        print(f"   ✅ {symbol}: {result.scanner_score:.1f} score - {result.setup_type.value}")
                    elif result:
                        print(f"   📊 {symbol}: {result.scanner_score:.1f} score (below threshold)")
                    else:
                        print(f"   ❌ {symbol}: No data")
                        
                    # Small delay to avoid overwhelming IB Gateway
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"   💥 {symbol}: Scan failed - {e}")
                    continue
            
            all_results.extend(sector_results)
            print(f"   📊 {len(sector_results)} opportunities found in {sector.value}")
        
        # Sort by scanner score
        all_results.sort(key=lambda x: x.scanner_score, reverse=True)
        
        print(f"\n🏆 TOTAL OPPORTUNITIES: {len(all_results)}")
        return all_results
    
    def display_results(self, results: List[ScannerResult]):
        """Display scan results"""
        if not results:
            print("\n❌ No trading opportunities found")
            return
        
        print(f"\n🏆 TOP TRADING OPPORTUNITIES")
        print("=" * 100)
        
        # Top 15 results
        top_results = results[:15]
        
        print(f"{'Rank':<4} {'Symbol':<8} {'Setup':<12} {'Score':<6} {'R/R':<6} {'Entry':<8} {'Growth':<10}")
        print("-" * 100)
        
        for i, result in enumerate(top_results, 1):
            print(f"{i:<4} {result.symbol:<8} {result.setup_type.value:<12} {result.scanner_score:<6.1f} "
                  f"{result.risk_reward_ratio:<6.1f} ${result.entry_price:<7.2f} {result.growth_potential:<10}")
        
        # Detailed analysis of top 5
        print(f"\n📊 DETAILED ANALYSIS - TOP 5 PICKS")
        print("=" * 80)
        
        for i, result in enumerate(top_results[:5], 1):
            print(f"\n🥇 #{i} {result.symbol} - {result.company_name}")
            print(f"   🎯 Sector: {result.sector.value}")
            print(f"   📈 Setup: {result.setup_type.value} (Score: {result.scanner_score:.1f})")
            print(f"   💰 Market Cap: ${result.market_cap/1e9:.1f}B")
            print(f"   📊 Price: ${result.current_price:.2f} ({result.price_change_1d:+.1f}% today)")
            print(f"   🎯 Entry: ${result.entry_price:.2f} | Stop: ${result.stop_loss:.2f} | Target: ${result.target_price:.2f}")
            print(f"   ⚖️  Risk/Reward: {result.risk_reward_ratio:.1f}:1")
            print(f"   📈 RSI: {result.rsi:.1f} | MACD: {result.macd_signal}")
            print(f"   🚀 Growth Potential: {result.growth_potential}")
            print(f"   🎪 Catalysts: {', '.join(result.catalyst_events)}")
        
        # Sector breakdown
        sector_breakdown = {}
        for result in results:
            sector = result.sector.value
            if sector not in sector_breakdown:
                sector_breakdown[sector] = []
            sector_breakdown[sector].append(result)
        
        print(f"\n📈 SECTOR BREAKDOWN")
        print("-" * 40)
        for sector, sector_results in sorted(sector_breakdown.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score = sum(r.scanner_score for r in sector_results) / len(sector_results)
            print(f"   {sector:<20}: {len(sector_results):2d} opportunities (avg: {avg_score:.1f})")

async def main():
    """Main scanner execution"""
    scanner = ProductionScanner()
    
    start_time = datetime.now()
    
    try:
        # Run scan
        results = await scanner.run_full_scan(min_score=60)
        
        # Display results
        scanner.display_results(results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Scan completed in {duration:.1f} seconds")
        print(f"🎯 {len(results)} trading opportunities identified")
        
        if results:
            top_pick = results[0]
            print(f"\n🏆 TOP PICK: {top_pick.symbol}")
            print(f"💡 {top_pick.setup_type.value} setup in {top_pick.sector.value}")
            print(f"🚀 {top_pick.growth_potential} growth potential!")
            print(f"📊 Scanner Score: {top_pick.scanner_score:.1f}/100")
        
    except Exception as e:
        logger.error(f"Scanner execution failed: {e}", "SCANNER")
        print(f"❌ Scanner failed: {e}")
    
    finally:
        # Cleanup
        if hasattr(scanner, 'ib_gateway'):
            scanner.ib_gateway.disconnect()  # Remove await - it's not async

if __name__ == "__main__":
    asyncio.run(main())
