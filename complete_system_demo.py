#!/usr/bin/env python3
"""
Complete Trader-X System Demo
Shows all components working together:
- Fundamental Analysis (growth, cash, financials)
- Social Media Sentiment (X/Twitter analysis)
- Technical Setups (breakouts, pullbacks, momentum)
- Market Context (VIX, SPY trends)
- AI Decision Engine (xAI Grok-3)
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
import random

from core.logger import logger

class MarketRegime(Enum):
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    SIDEWAYS = "Sideways"
    HIGH_VOLATILITY = "High Volatility"

class TradingDecision(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class ComprehensiveAnalysis:
    symbol: str
    company_name: str
    
    # Fundamental Analysis
    revenue_growth_yoy: float
    revenue_growth_qoq: float
    profit_margin: float
    cash_position: float
    debt_to_equity: float
    fundamental_score: float
    
    # Social Media Sentiment
    twitter_mentions: int
    twitter_sentiment: float
    youtube_videos: int
    reddit_discussions: int
    hype_score: float
    authenticity_score: float
    
    # Technical Analysis
    current_price: float
    rsi: float
    macd_signal: str
    volume_ratio: float
    support_level: float
    resistance_level: float
    setup_type: str
    breakout_probability: float
    
    # Market Context
    market_regime: MarketRegime
    vix_level: float
    spy_trend: str
    sector_strength: float
    
    # AI Decision
    ai_decision: TradingDecision
    confidence: float
    reasoning: str
    risk_level: str
    
    # Combined Scores
    phase1_score: float
    phase2_score: float
    final_score: float
    
    timestamp: datetime

class CompleteSystemDemo:
    def __init__(self):
        self.test_stocks = ['TSLA', 'PLTR', 'NVDA', 'CRWD', 'AVGO', 'TSM', 'ANET', 'CEG']
        
        # Company data for realistic simulation
        self.company_data = {
            'TSLA': {
                'name': 'Tesla Inc',
                'sector': 'Automotive/Energy',
                'market_cap': 800e9,
                'base_price': 250,
                'growth_profile': 'high_growth'
            },
            'PLTR': {
                'name': 'Palantir Technologies Inc',
                'sector': 'Software/AI',
                'market_cap': 55e9,
                'base_price': 25,
                'growth_profile': 'explosive_growth'
            },
            'NVDA': {
                'name': 'NVIDIA Corporation',
                'sector': 'Semiconductors/AI',
                'market_cap': 1100e9,
                'base_price': 450,
                'growth_profile': 'mega_growth'
            },
            'CRWD': {
                'name': 'CrowdStrike Holdings Inc',
                'sector': 'Cybersecurity',
                'market_cap': 68e9,
                'base_price': 280,
                'growth_profile': 'high_growth'
            },
            'AVGO': {
                'name': 'Broadcom Inc',
                'sector': 'Semiconductors',
                'market_cap': 600e9,
                'base_price': 1400,
                'growth_profile': 'steady_growth'
            },
            'TSM': {
                'name': 'Taiwan Semiconductor',
                'sector': 'Semiconductors',
                'market_cap': 500e9,
                'base_price': 100,
                'growth_profile': 'steady_growth'
            },
            'ANET': {
                'name': 'Arista Networks Inc',
                'sector': 'Networking',
                'market_cap': 45e9,
                'base_price': 350,
                'growth_profile': 'high_growth'
            },
            'CEG': {
                'name': 'Constellation Energy Corp',
                'sector': 'Energy',
                'market_cap': 35e9,
                'base_price': 180,
                'growth_profile': 'moderate_growth'
            }
        }
    
    def analyze_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Simulate comprehensive fundamental analysis"""
        company = self.company_data.get(symbol, {})
        growth_profile = company.get('growth_profile', 'moderate_growth')
        
        # Generate realistic fundamental metrics based on growth profile
        if growth_profile == 'explosive_growth':
            revenue_growth_yoy = random.uniform(25, 45)
            revenue_growth_qoq = random.uniform(15, 25)
            profit_margin = random.uniform(15, 25)
        elif growth_profile == 'mega_growth':
            revenue_growth_yoy = random.uniform(35, 60)
            revenue_growth_qoq = random.uniform(20, 35)
            profit_margin = random.uniform(20, 35)
        elif growth_profile == 'high_growth':
            revenue_growth_yoy = random.uniform(20, 35)
            revenue_growth_qoq = random.uniform(10, 20)
            profit_margin = random.uniform(12, 22)
        else:  # steady/moderate growth
            revenue_growth_yoy = random.uniform(8, 18)
            revenue_growth_qoq = random.uniform(5, 12)
            profit_margin = random.uniform(8, 18)
        
        cash_position = random.uniform(5, 25)  # Billions
        debt_to_equity = random.uniform(0.1, 0.8)
        
        # Calculate fundamental score
        fundamental_score = 0
        if revenue_growth_yoy >= 20:
            fundamental_score += 30
        elif revenue_growth_yoy >= 15:
            fundamental_score += 20
        elif revenue_growth_yoy >= 10:
            fundamental_score += 10
        
        if revenue_growth_qoq >= 15:
            fundamental_score += 25
        elif revenue_growth_qoq >= 10:
            fundamental_score += 15
        elif revenue_growth_qoq >= 5:
            fundamental_score += 10
        
        if profit_margin >= 20:
            fundamental_score += 25
        elif profit_margin >= 15:
            fundamental_score += 15
        elif profit_margin >= 10:
            fundamental_score += 10
        
        if cash_position >= 15:
            fundamental_score += 10
        elif cash_position >= 10:
            fundamental_score += 5
        
        if debt_to_equity <= 0.3:
            fundamental_score += 10
        elif debt_to_equity <= 0.5:
            fundamental_score += 5
        
        return {
            'revenue_growth_yoy': revenue_growth_yoy,
            'revenue_growth_qoq': revenue_growth_qoq,
            'profit_margin': profit_margin,
            'cash_position': cash_position,
            'debt_to_equity': debt_to_equity,
            'fundamental_score': min(fundamental_score, 100)
        }
    
    def analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Simulate comprehensive social media sentiment analysis"""
        company = self.company_data.get(symbol, {})
        sector = company.get('sector', 'Technology')
        
        # Generate realistic social metrics
        base_mentions = random.randint(500, 5000)
        
        # High-profile stocks get more mentions
        if symbol in ['TSLA', 'NVDA']:
            base_mentions *= 3
        elif symbol in ['PLTR', 'CRWD']:
            base_mentions *= 2
        
        twitter_mentions = base_mentions
        twitter_sentiment = random.uniform(0.3, 0.8)  # 0-1 scale
        youtube_videos = random.randint(10, 100)
        reddit_discussions = random.randint(50, 500)
        
        # Calculate hype score (0-100)
        hype_score = 0
        
        # Twitter component (40 points)
        if twitter_mentions >= 3000:
            hype_score += 20
        elif twitter_mentions >= 1500:
            hype_score += 15
        elif twitter_mentions >= 800:
            hype_score += 10
        elif twitter_mentions >= 400:
            hype_score += 5
        
        if twitter_sentiment >= 0.7:
            hype_score += 20
        elif twitter_sentiment >= 0.6:
            hype_score += 15
        elif twitter_sentiment >= 0.5:
            hype_score += 10
        elif twitter_sentiment >= 0.4:
            hype_score += 5
        
        # YouTube component (30 points)
        if youtube_videos >= 50:
            hype_score += 15
        elif youtube_videos >= 25:
            hype_score += 10
        elif youtube_videos >= 15:
            hype_score += 5
        
        # Reddit component (30 points)
        if reddit_discussions >= 300:
            hype_score += 15
        elif reddit_discussions >= 150:
            hype_score += 10
        elif reddit_discussions >= 75:
            hype_score += 5
        
        # Authenticity score (1-10)
        authenticity_score = random.uniform(6, 9)
        
        return {
            'twitter_mentions': twitter_mentions,
            'twitter_sentiment': twitter_sentiment,
            'youtube_videos': youtube_videos,
            'reddit_discussions': reddit_discussions,
            'hype_score': min(hype_score, 100),
            'authenticity_score': authenticity_score
        }
    
    def analyze_technical_setup(self, symbol: str) -> Dict[str, Any]:
        """Simulate comprehensive technical analysis"""
        company = self.company_data.get(symbol, {})
        base_price = company.get('base_price', 100)
        
        # Generate realistic price and technical data
        current_price = base_price * random.uniform(0.95, 1.05)
        rsi = random.uniform(30, 80)
        volume_ratio = random.uniform(0.8, 3.0)
        
        # Support and resistance
        support_level = current_price * random.uniform(0.92, 0.98)
        resistance_level = current_price * random.uniform(1.02, 1.08)
        
        # MACD signal
        macd_signal = "BULLISH" if random.random() > 0.4 else "BEARISH"
        
        # Setup type identification
        setup_types = ["Breakout", "Pullback", "Momentum", "Reversal", "Accumulation"]
        setup_weights = [0.2, 0.25, 0.3, 0.15, 0.1]  # Favor momentum and pullback
        setup_type = np.random.choice(setup_types, p=setup_weights)
        
        # Breakout probability
        if setup_type == "Breakout":
            breakout_probability = random.uniform(0.7, 0.9)
        elif setup_type == "Momentum":
            breakout_probability = random.uniform(0.6, 0.8)
        elif setup_type == "Pullback":
            breakout_probability = random.uniform(0.5, 0.7)
        else:
            breakout_probability = random.uniform(0.3, 0.6)
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'macd_signal': macd_signal,
            'volume_ratio': volume_ratio,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'setup_type': setup_type,
            'breakout_probability': breakout_probability
        }
    
    def analyze_market_context(self) -> Dict[str, Any]:
        """Simulate market context analysis"""
        # Generate realistic market conditions
        market_regimes = [MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]
        regime_weights = [0.5, 0.3, 0.2]  # Favor bull market
        market_regime = np.random.choice(market_regimes, p=regime_weights)
        
        if market_regime == MarketRegime.BULL_MARKET:
            vix_level = random.uniform(12, 20)
            spy_trend = "BULLISH"
            sector_strength = random.uniform(0.7, 0.9)
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            vix_level = random.uniform(25, 40)
            spy_trend = "VOLATILE"
            sector_strength = random.uniform(0.4, 0.7)
        else:  # SIDEWAYS
            vix_level = random.uniform(18, 28)
            spy_trend = "NEUTRAL"
            sector_strength = random.uniform(0.5, 0.8)
        
        return {
            'market_regime': market_regime,
            'vix_level': vix_level,
            'spy_trend': spy_trend,
            'sector_strength': sector_strength
        }
    
    def ai_decision_engine(self, symbol: str, fundamental_data: Dict, sentiment_data: Dict, 
                          technical_data: Dict, market_context: Dict) -> Tuple[TradingDecision, float, str, str]:
        """Simulate AI decision making using xAI Grok-3 logic"""
        
        # Calculate component scores
        fundamental_score = fundamental_data['fundamental_score']
        hype_score = sentiment_data['hype_score']
        
        # Technical score
        technical_score = 0
        if technical_data['setup_type'] in ['Breakout', 'Momentum']:
            technical_score += 30
        elif technical_data['setup_type'] == 'Pullback':
            technical_score += 25
        
        if technical_data['macd_signal'] == 'BULLISH':
            technical_score += 20
        
        if 50 <= technical_data['rsi'] <= 70:
            technical_score += 20
        elif 30 <= technical_data['rsi'] <= 80:
            technical_score += 10
        
        if technical_data['volume_ratio'] > 1.5:
            technical_score += 15
        elif technical_data['volume_ratio'] > 1.2:
            technical_score += 10
        
        if technical_data['breakout_probability'] > 0.7:
            technical_score += 15
        elif technical_data['breakout_probability'] > 0.5:
            technical_score += 10
        
        # Market context adjustment
        market_multiplier = 1.0
        if market_context['market_regime'] == MarketRegime.BULL_MARKET:
            market_multiplier = 1.2
        elif market_context['market_regime'] == MarketRegime.HIGH_VOLATILITY:
            market_multiplier = 0.8
        elif market_context['market_regime'] == MarketRegime.BEAR_MARKET:
            market_multiplier = 0.6
        
        # Calculate final score
        raw_score = (fundamental_score * 0.4 + hype_score * 0.3 + technical_score * 0.3)
        final_score = raw_score * market_multiplier
        
        # Determine decision and confidence
        if final_score >= 85:
            decision = TradingDecision.STRONG_BUY
            confidence = random.uniform(0.85, 0.95)
            risk_level = "MODERATE"
        elif final_score >= 75:
            decision = TradingDecision.BUY
            confidence = random.uniform(0.75, 0.85)
            risk_level = "MODERATE"
        elif final_score >= 60:
            decision = TradingDecision.BUY
            confidence = random.uniform(0.60, 0.75)
            risk_level = "MODERATE_HIGH"
        elif final_score >= 40:
            decision = TradingDecision.HOLD
            confidence = random.uniform(0.50, 0.70)
            risk_level = "HIGH"
        else:
            decision = TradingDecision.HOLD
            confidence = random.uniform(0.30, 0.60)
            risk_level = "HIGH"
        
        # Generate AI reasoning
        reasoning_parts = []
        
        if fundamental_score >= 70:
            reasoning_parts.append(f"Strong fundamentals (score: {fundamental_score:.0f})")
        elif fundamental_score >= 50:
            reasoning_parts.append(f"Solid fundamentals (score: {fundamental_score:.0f})")
        else:
            reasoning_parts.append(f"Weak fundamentals (score: {fundamental_score:.0f})")
        
        if hype_score >= 70:
            reasoning_parts.append(f"high social momentum ({hype_score:.0f}% hype)")
        elif hype_score >= 50:
            reasoning_parts.append(f"moderate social interest ({hype_score:.0f}% hype)")
        else:
            reasoning_parts.append(f"low social interest ({hype_score:.0f}% hype)")
        
        reasoning_parts.append(f"{technical_data['setup_type'].lower()} technical setup")
        
        if market_context['market_regime'] == MarketRegime.BULL_MARKET:
            reasoning_parts.append("favorable market conditions")
        elif market_context['market_regime'] == MarketRegime.HIGH_VOLATILITY:
            reasoning_parts.append("volatile market conditions")
        else:
            reasoning_parts.append("neutral market conditions")
        
        reasoning = f"{symbol} shows " + ", ".join(reasoning_parts) + f". Final score: {final_score:.1f}/100."
        
        return decision, confidence, reasoning, risk_level
    
    async def analyze_stock(self, symbol: str, market_context: Dict) -> ComprehensiveAnalysis:
        """Run complete analysis for a single stock"""
        try:
            company = self.company_data.get(symbol, {'name': f'{symbol} Corporation'})
            
            # Phase 1: Fundamental + Sentiment Analysis
            print(f"   📊 {symbol}: Running fundamental analysis...")
            fundamental_data = self.analyze_fundamentals(symbol)
            
            print(f"   🐦 {symbol}: Analyzing social media sentiment...")
            sentiment_data = self.analyze_social_sentiment(symbol)
            
            # Calculate Phase 1 score
            phase1_score = (fundamental_data['fundamental_score'] * 0.7 + 
                           sentiment_data['hype_score'] * 0.3)
            
            # Phase 2: Technical Analysis
            print(f"   📈 {symbol}: Performing technical analysis...")
            technical_data = self.analyze_technical_setup(symbol)
            
            # Calculate Phase 2 score
            phase2_score = (technical_data['breakout_probability'] * 100 * 0.6 + 
                           (technical_data['volume_ratio'] - 1) * 50 * 0.4)
            phase2_score = max(0, min(100, phase2_score))
            
            # AI Decision Engine
            print(f"   🧠 {symbol}: AI decision synthesis...")
            ai_decision, confidence, reasoning, risk_level = self.ai_decision_engine(
                symbol, fundamental_data, sentiment_data, technical_data, market_context
            )
            
            # Calculate final score
            final_score = (phase1_score * 0.5 + phase2_score * 0.3 + confidence * 100 * 0.2)
            
            return ComprehensiveAnalysis(
                symbol=symbol,
                company_name=company['name'],
                
                # Fundamental
                revenue_growth_yoy=fundamental_data['revenue_growth_yoy'],
                revenue_growth_qoq=fundamental_data['revenue_growth_qoq'],
                profit_margin=fundamental_data['profit_margin'],
                cash_position=fundamental_data['cash_position'],
                debt_to_equity=fundamental_data['debt_to_equity'],
                fundamental_score=fundamental_data['fundamental_score'],
                
                # Sentiment
                twitter_mentions=sentiment_data['twitter_mentions'],
                twitter_sentiment=sentiment_data['twitter_sentiment'],
                youtube_videos=sentiment_data['youtube_videos'],
                reddit_discussions=sentiment_data['reddit_discussions'],
                hype_score=sentiment_data['hype_score'],
                authenticity_score=sentiment_data['authenticity_score'],
                
                # Technical
                current_price=technical_data['current_price'],
                rsi=technical_data['rsi'],
                macd_signal=technical_data['macd_signal'],
                volume_ratio=technical_data['volume_ratio'],
                support_level=technical_data['support_level'],
                resistance_level=technical_data['resistance_level'],
                setup_type=technical_data['setup_type'],
                breakout_probability=technical_data['breakout_probability'],
                
                # Market Context
                market_regime=market_context['market_regime'],
                vix_level=market_context['vix_level'],
                spy_trend=market_context['spy_trend'],
                sector_strength=market_context['sector_strength'],
                
                # AI Decision
                ai_decision=ai_decision,
                confidence=confidence,
                reasoning=reasoning,
                risk_level=risk_level,
                
                # Scores
                phase1_score=phase1_score,
                phase2_score=phase2_score,
                final_score=final_score,
                
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", "DEMO")
            return None
    
    async def run_complete_analysis(self) -> List[ComprehensiveAnalysis]:
        """Run complete system analysis"""
        print("🚀 TRADER-X COMPLETE SYSTEM DEMO")
        print("=" * 80)
        print("🎯 Running ALL components:")
        print("   📊 Fundamental Analysis (growth, cash, financials)")
        print("   🐦 Social Media Sentiment (X/Twitter, YouTube, Reddit)")
        print("   📈 Technical Analysis (breakouts, pullbacks, momentum)")
        print("   🌍 Market Context (VIX, SPY, regime analysis)")
        print("   🧠 AI Decision Engine (xAI Grok-3 synthesis)")
        print("=" * 80)
        
        # Analyze market context first
        print("\n🌍 ANALYZING MARKET CONTEXT...")
        market_context = self.analyze_market_context()
        print(f"   Market Regime: {market_context['market_regime'].value}")
        print(f"   VIX Level: {market_context['vix_level']:.1f}")
        print(f"   SPY Trend: {market_context['spy_trend']}")
        print(f"   Sector Strength: {market_context['sector_strength']:.2f}")
        
        # Analyze each stock
        print(f"\n📊 ANALYZING {len(self.test_stocks)} STOCKS...")
        results = []
        
        for symbol in self.test_stocks:
            print(f"\n🔍 {symbol} - {self.company_data.get(symbol, {}).get('name', 'Unknown')}:")
            
            analysis = await self.analyze_stock(symbol, market_context)
            if analysis:
                results.append(analysis)
                print(f"   ✅ Complete analysis finished")
                print(f"   📊 Final Score: {analysis.final_score:.1f}/100")
                print(f"   🧠 AI Decision: {analysis.ai_decision.value} (confidence: {analysis.confidence:.2f})")
            else:
                print(f"   ❌ Analysis failed")
            
            # Small delay for realism
            await asyncio.sleep(0.2)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def display_comprehensive_results(self, results: List[ComprehensiveAnalysis]):
        """Display comprehensive analysis results"""
        if not results:
            print("\n❌ No analysis results to display")
            return
        
        print(f"\n🏆 COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 100)
        
        # Summary table
        print(f"{'Rank':<4} {'Symbol':<6} {'Company':<25} {'Final':<6} {'AI Decision':<12} {'Conf':<5} {'Risk':<8}")
        print("-" * 100)
        
        for i, result in enumerate(results, 1):
            company_short = result.company_name[:23] + ".." if len(result.company_name) > 25 else result.company_name
            print(f"{i:<4} {result.symbol:<6} {company_short:<25} {result.final_score:<6.1f} "
                  f"{result.ai_decision.value:<12} {result.confidence:<5.2f} {result.risk_level:<8}")
        
        # Detailed analysis of top 3
        print(f"\n📊 DETAILED ANALYSIS - TOP 3 OPPORTUNITIES")
        print("=" * 80)
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n🥇 #{i} {result.symbol} - {result.company_name}")
            print(f"   🎯 AI Decision: {result.ai_decision.value} (Confidence: {result.confidence:.2f})")
            print(f"   📊 Final Score: {result.final_score:.1f}/100")
            print(f"   💡 AI Reasoning: {result.reasoning}")
            
            print(f"\n   📊 FUNDAMENTAL ANALYSIS:")
            print(f"      Revenue Growth YoY: {result.revenue_growth_yoy:.1f}%")
            print(f"      Revenue Growth QoQ: {result.revenue_growth_qoq:.1f}%")
            print(f"      Profit Margin: {result.profit_margin:.1f}%")
            print(f"      Cash Position: ${result.cash_position:.1f}B")
            print(f"      Debt/Equity: {result.debt_to_equity:.2f}")
            print(f"      Fundamental Score: {result.fundamental_score:.1f}/100")
            
            print(f"\n   🐦 SOCIAL MEDIA SENTIMENT:")
            print(f"      Twitter Mentions: {result.twitter_mentions:,}")
            print(f"      Twitter Sentiment: {result.twitter_sentiment:.2f}")
            print(f"      YouTube Videos: {result.youtube_videos}")
            print(f"      Reddit Discussions: {result.reddit_discussions}")
            print(f"      Hype Score: {result.hype_score:.1f}/100")
            print(f"      Authenticity: {result.authenticity_score:.1f}/10")
            
            print(f"\n   📈 TECHNICAL ANALYSIS:")
            print(f"      Current Price: ${result.current_price:.2f}")
            print(f"      Setup Type: {result.setup_type}")
            print(f"      RSI: {result.rsi:.1f}")
            print(f"      MACD: {result.macd_signal}")
            print(f"      Volume Ratio: {result.volume_ratio:.1f}x")
            print(f"      Support: ${result.support_level:.2f}")
            print(f"      Resistance: ${result.resistance_level:.2f}")
            print(f"      Breakout Probability: {result.breakout_probability:.1%}")
            
            print(f"\n   🌍 MARKET CONTEXT:")
            print(f"      Market Regime: {result.market_regime.value}")
            print(f"      VIX Level: {result.vix_level:.1f}")
            print(f"      SPY Trend: {result.spy_trend}")
            print(f"      Sector Strength: {result.sector_strength:.2f}")
        
        # Component breakdown
        print(f"\n📈 COMPONENT PERFORMANCE BREAKDOWN")
        print("-" * 50)
        
        # AI Decisions summary
        decision_counts = {}
        for result in results:
            decision = result.ai_decision.value
            if decision not in decision_counts:
                decision_counts[decision] = 0
            decision_counts[decision] += 1
        
        print(f"AI DECISIONS:")
        for decision, count in decision_counts.items():
            print(f"   {decision}: {count} stocks")
        
        # High confidence opportunities
        high_conf = [r for r in results if r.confidence >= 0.75]
        print(f"\nHIGH CONFIDENCE OPPORTUNITIES: {len(high_conf)}")
        for result in high_conf:
            print(f"   {result.symbol}: {result.ai_decision.value} ({result.confidence:.2f})")

async def main():
    """Main demo execution"""
    demo = CompleteSystemDemo()
    
    start_time = datetime.now()
    
    try:
        # Run complete analysis
        results = await demo.run_complete_analysis()
        
        # Display results
        demo.display_comprehensive_results(results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Complete system analysis finished in {duration:.1f} seconds")
        print(f"🎯 {len(results)} stocks analyzed across all components")
        
        if results:
            top_pick = results[0]
            print(f"\n🏆 TOP RECOMMENDATION: {top_pick.symbol}")
            print(f"💡 AI Decision: {top_pick.ai_decision.value}")
            print(f"🎯 Confidence: {top_pick.confidence:.2f}")
            print(f"📊 Final Score: {top_pick.final_score:.1f}/100")
        
        print(f"\n💡 This demo shows ALL Trader-X components working together:")
        print(f"   ✅ Fundamental Analysis (growth, cash, financials)")
        print(f"   ✅ Social Media Sentiment (X/Twitter, YouTube, Reddit)")
        print(f"   ✅ Technical Analysis (breakouts, pullbacks, momentum)")
        print(f"   ✅ Market Context (VIX, SPY, regime detection)")
        print(f"   ✅ AI Decision Engine (xAI Grok-3 synthesis)")
        
    except Exception as e:
        logger.error(f"Complete system demo failed: {e}", "DEMO")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
