#!/usr/bin/env python3
"""
Robust Production Market Scanner
Uses multiple data sources with intelligent fallback
Works 24/7 regardless of market hours or data source issues
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
import yfinance as yf

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
    data_source: str
    scan_timestamp: datetime

class RobustScanner:
    def __init__(self):
        self.ib_gateway = None
        self.ib_connected = False
        self.config = TradingConfig()
        self.scan_results = []
        self.rate_limit_delay = 0.5  # Delay between requests to avoid rate limits
        
        # Define our universe of potential winners
        self.growth_universe = {
            MarketSector.TECHNOLOGY: [
                # AI & Cloud (Top Priority)
                'PLTR', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'PANW',
                'MDB', 'ESTC', 'FSLY', 'TWLO', 'DOCN', 'TEAM', 'ZM',
                
                # Semiconductors
                'NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL', 'AMAT', 'LRCX', 'KLAC',
                
                # Software
                'CRM', 'NOW', 'VEEV', 'WDAY', 'ADBE', 'ORCL', 'MSFT'
            ],
            
            MarketSector.HEALTHCARE: [
                'MRNA', 'BNTX', 'NVAX', 'CRSP', 'EDIT', 'NTLA', 'BEAM',
                'ILMN', 'PACB', 'TDOC', 'DXCM', 'ISRG'
            ],
            
            MarketSector.FINANCIAL: [
                'SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'LC',
                'BILL', 'PAYO', 'NU', 'MELI', 'SE'
            ],
            
            MarketSector.ENERGY: [
                'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'PLUG', 'BE', 'BLDP',
                'FCEL', 'CHPT', 'BLNK', 'QS', 'STEM'
            ],
            
            MarketSector.CONSUMER: [
                'SHOP', 'ROKU', 'UBER', 'DASH', 'ABNB', 'PTON', 'NFLX'
            ],
            
            MarketSector.INDUSTRIAL: [
                'IRBT', 'ZBRA', 'TER', 'OMCL'
            ]
        }
        
        # Company names for better display
        self.company_names = {
            'PLTR': 'Palantir Technologies Inc',
            'SNOW': 'Snowflake Inc',
            'DDOG': 'Datadog Inc',
            'NET': 'Cloudflare Inc',
            'CRWD': 'CrowdStrike Holdings Inc',
            'ZS': 'Zscaler Inc',
            'OKTA': 'Okta Inc',
            'PANW': 'Palo Alto Networks Inc',
            'MDB': 'MongoDB Inc',
            'NVDA': 'NVIDIA Corporation',
            'AMD': 'Advanced Micro Devices Inc',
            'MRNA': 'Moderna Inc',
            'COIN': 'Coinbase Global Inc',
            'ENPH': 'Enphase Energy Inc',
            'SHOP': 'Shopify Inc',
            'IRBT': 'iRobot Corporation'
        }
    
    async def initialize_connections(self) -> Dict[str, bool]:
        """Initialize all data connections"""
        connections = {
            'ib_gateway': False,
            'yfinance': True  # Always available
        }
        
        # Try IB Gateway connection
        try:
            print("🔌 Attempting IB Gateway connection...")
            self.ib_gateway = IBGatewayConnector(port=4002)
            self.ib_connected = self.ib_gateway.connect()
            connections['ib_gateway'] = self.ib_connected
            
            if self.ib_connected:
                print("✅ IB Gateway connected successfully")
            else:
                print("⚠️  IB Gateway not available - using fallback data sources")
                
        except Exception as e:
            print(f"⚠️  IB Gateway connection failed: {e}")
            logger.warning(f"IB Gateway connection failed: {e}", "SCANNER")
        
        return connections
    
    def get_market_data_robust(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data with robust fallback system"""
        data_source = "UNKNOWN"
        
        # Method 1: Try IB Gateway first (if connected and markets open)
        if self.ib_connected:
            try:
                # Quick timeout for IB Gateway to avoid long waits
                quote = self.ib_gateway.get_market_data(symbol, timeout=3)
                hist_data = self.ib_gateway.get_historical_data(symbol, duration="30 D", bar_size="1 day", timeout=5)
                
                if quote and hist_data is not None and len(hist_data) >= 20:
                    data_source = "IB_GATEWAY"
                    return {
                        'quote': quote,
                        'historical': hist_data,
                        'symbol': symbol,
                        'data_source': data_source
                    }
            except Exception as e:
                logger.debug(f"IB Gateway data fetch failed for {symbol}: {e}", "SCANNER")
        
        # Method 2: Fallback to yfinance with rate limiting
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")  # Get 6 months of data
            
            if len(hist) < 20:
                return None
            
            # Convert yfinance data to our format
            hist_data = pd.DataFrame({
                'open': hist['Open'],
                'high': hist['High'], 
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume']
            })
            
            # Create quote from latest data and info
            quote = {
                'last': hist['Close'].iloc[-1],
                'bid': hist['Close'].iloc[-1] * 0.999,  # Approximate bid
                'ask': hist['Close'].iloc[-1] * 1.001,  # Approximate ask
                'volume': hist['Volume'].iloc[-1],
                'marketCap': info.get('marketCap', 0),
                'companyName': info.get('longName', self.company_names.get(symbol, symbol))
            }
            
            data_source = "YFINANCE"
            return {
                'quote': quote,
                'historical': hist_data,
                'symbol': symbol,
                'data_source': data_source
            }
            
        except Exception as e:
            logger.debug(f"YFinance data fetch failed for {symbol}: {e}", "SCANNER")
            # Increase rate limit delay if we hit limits
            if "429" in str(e) or "Too Many Requests" in str(e):
                self.rate_limit_delay = min(self.rate_limit_delay * 1.5, 5.0)
                logger.warning(f"Rate limit hit, increasing delay to {self.rate_limit_delay}s", "SCANNER")
        
        return None
    
    def calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            # Ensure we have the right column names
            if 'Close' in hist_data.columns:
                close = hist_data['Close']
                high = hist_data['High']
                low = hist_data['Low']
                volume = hist_data['Volume']
            else:
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
        
        # Add earnings catalyst
        catalysts.append("Upcoming earnings")
        
        return catalysts[:3]  # Top 3 catalysts
    
    async def scan_symbol(self, symbol: str, sector: MarketSector) -> Optional[ScannerResult]:
        """Scan individual symbol for trading opportunities"""
        try:
            # Get market data with robust fallback
            market_data = self.get_market_data_robust(symbol)
            if not market_data:
                return None
            
            quote = market_data['quote']
            hist_data = market_data['historical']
            data_source = market_data['data_source']
            
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
            
            # Get market cap
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
                company_name=quote.get('companyName', self.company_names.get(symbol, symbol)),
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
                data_source=data_source,
                scan_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Symbol scan failed for {symbol}: {e}", "SCANNER")
            return None
    
    async def run_full_scan(self, min_score: float = 60) -> List[ScannerResult]:
        """Run complete market scan with robust data handling"""
        print("🚀 ROBUST PRODUCTION MARKET SCANNER")
        print("=" * 60)
        print("🎯 Finding next PLTR, CRWD, NVDA")
        print(f"📊 Minimum Score: {min_score}")
        print("🔄 Multi-source data with intelligent fallback")
        print("=" * 60)
        
        # Initialize connections
        connections = await self.initialize_connections()
        print(f"📡 Data Sources: IB Gateway: {connections['ib_gateway']}, YFinance: {connections['yfinance']}")
        
        all_results = []
        
        for sector, symbols in self.growth_universe.items():
            print(f"\n🔍 Scanning {sector.value} ({len(symbols)} symbols)...")
            sector_results = []
            
            for symbol in symbols:
                try:
                    result = await self.scan_symbol(symbol, sector)
                    if result and result.scanner_score >= min_score:
                        sector_results.append(result)
                        print(f"   ✅ {symbol}: {result.scanner_score:.1f} score - {result.setup_type.value} ({result.data_source})")
                    elif result:
                        print(f"   📊 {symbol}: {result.scanner_score:.1f} score ({result.data_source}) - below {min_score}")
                    else:
                        print(f"   ❌ {symbol}: No data available")
                        
                    # Small delay between requests
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
        print("=" * 120)
        
        # Top 15 results
        top_results = results[:15]
        
        print(f"{'Rank':<4} {'Symbol':<8} {'Setup':<12} {'Score':<6} {'R/R':<6} {'Entry':<8} {'Growth':<10} {'Source':<10}")
        print("-" * 120)
        
        for i, result in enumerate(top_results, 1):
            print(f"{i:<4} {result.symbol:<8} {result.setup_type.value:<12} {result.scanner_score:<6.1f} "
                  f"{result.risk_reward_ratio:<6.1f} ${result.entry_price:<7.2f} {result.growth_potential:<10} {result.data_source:<10}")
        
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
            print(f"   📡 Data Source: {result.data_source}")
        
        # Data source breakdown
        source_breakdown = {}
        for result in results:
            source = result.data_source
            if source not in source_breakdown:
                source_breakdown[source] = 0
            source_breakdown[source] += 1
        
        print(f"\n📡 DATA SOURCE BREAKDOWN")
        print("-" * 30)
        for source, count in source_breakdown.items():
            print(f"   {source:<15}: {count:2d} symbols")

async def main():
    """Main scanner execution"""
    scanner = RobustScanner()
    
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
            print(f"📡 Data Source: {top_pick.data_source}")
        
    except Exception as e:
        logger.error(f"Scanner execution failed: {e}", "SCANNER")
        print(f"❌ Scanner failed: {e}")
    
    finally:
        # Cleanup
        if hasattr(scanner, 'ib_gateway') and scanner.ib_gateway:
            scanner.ib_gateway.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
