#!/usr/bin/env python3
"""
Next Big Winner Scanner
Identifies potential next PLTR, CRWD, NVDA before they explode
Focus: Early-stage disruptive companies with massive growth potential
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

class DisruptiveTheme(Enum):
    AI_ML = "Artificial Intelligence & Machine Learning"
    CYBERSECURITY = "Cybersecurity & Data Protection"
    CLOUD_INFRASTRUCTURE = "Cloud Infrastructure & SaaS"
    FINTECH = "Financial Technology"
    BIOTECH = "Biotechnology & Healthcare"
    CLEAN_ENERGY = "Clean Energy & Sustainability"
    AUTONOMOUS_VEHICLES = "Autonomous Vehicles & Mobility"
    QUANTUM_COMPUTING = "Quantum Computing"
    SPACE_TECH = "Space Technology"
    ROBOTICS = "Robotics & Automation"
    BLOCKCHAIN = "Blockchain & Crypto"
    EDGE_COMPUTING = "Edge Computing & IoT"

@dataclass
class WinnerCandidate:
    symbol: str
    company_name: str
    disruptive_theme: DisruptiveTheme
    market_cap: float
    revenue_growth_yoy: float
    revenue_growth_qoq: float
    gross_margin: float
    r_and_d_intensity: float
    insider_ownership: float
    institutional_ownership: float
    short_interest: float
    price_momentum_score: float
    volume_accumulation_score: float
    innovation_score: float
    market_opportunity_score: float
    competitive_moat_score: float
    execution_score: float
    total_winner_score: float
    risk_level: str
    catalyst_potential: str
    scan_timestamp: datetime

class NextWinnerScanner:
    def __init__(self):
        self.disruptive_universe = self._build_disruptive_universe()
        
    def _build_disruptive_universe(self) -> Dict[DisruptiveTheme, List[str]]:
        """Build universe of potentially disruptive companies by theme"""
        return {
            DisruptiveTheme.AI_ML: [
                'PLTR', 'C3AI', 'AI', 'SOUN', 'BBAI', 'UPST', 'LMND', 'PATH',
                'SNOW', 'DDOG', 'NET', 'ESTC', 'SPLK', 'VEEV', 'NOW', 'CRM',
                'NVDA', 'AMD', 'INTC', 'GOOGL', 'MSFT', 'META', 'TSLA'
            ],
            
            DisruptiveTheme.CYBERSECURITY: [
                'CRWD', 'ZS', 'OKTA', 'PANW', 'FTNT', 'CYBR', 'S', 'TENB',
                'QLYS', 'VRNS', 'SAIL', 'RBRK', 'CHKP', 'FEYE', 'PFPT'
            ],
            
            DisruptiveTheme.CLOUD_INFRASTRUCTURE: [
                'SNOW', 'DDOG', 'NET', 'FSLY', 'ESTC', 'MDB', 'DOCN', 'TEAM',
                'ATLASSIAN', 'ZM', 'TWLO', 'SEND', 'PD', 'WORK', 'FIVN'
            ],
            
            DisruptiveTheme.FINTECH: [
                'SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'LC',
                'OPEN', 'BILL', 'PAYO', 'STNE', 'NU', 'MELI', 'SE'
            ],
            
            DisruptiveTheme.BIOTECH: [
                'MRNA', 'BNTX', 'NVAX', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV',
                'ARCT', 'SANA', 'BLUE', 'FATE', 'CGEM', 'PACB', 'ILMN'
            ],
            
            DisruptiveTheme.CLEAN_ENERGY: [
                'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'NOVA', 'CSIQ', 'JKS',
                'PLUG', 'BE', 'BLDP', 'FCEL', 'CHPT', 'BLNK', 'QS', 'STEM'
            ],
            
            DisruptiveTheme.AUTONOMOUS_VEHICLES: [
                'TSLA', 'RIVN', 'LCID', 'NKLA', 'RIDE', 'GOEV', 'HYLN', 'WKHS',
                'LAZR', 'VLDR', 'LIDR', 'OUST', 'MVIS', 'INVZ', 'AEYE'
            ],
            
            DisruptiveTheme.SPACE_TECH: [
                'SPCE', 'RKLB', 'ASTR', 'IRDM', 'MAXR', 'SATS', 'GSAT', 'ORBC'
            ],
            
            DisruptiveTheme.ROBOTICS: [
                'IRBT', 'ISRG', 'ABB', 'FANUY', 'KUKA', 'OMCL', 'ZBRA', 'TER'
            ],
            
            DisruptiveTheme.BLOCKCHAIN: [
                'COIN', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN', 'MSTR', 'SQ'
            ],
            
            DisruptiveTheme.QUANTUM_COMPUTING: [
                'IBM', 'GOOGL', 'MSFT', 'IONQ', 'RGTI', 'QUBT', 'ARQQ'
            ]
        }
    
    def analyze_winner_potential(self, symbol: str, theme: DisruptiveTheme) -> Optional[WinnerCandidate]:
        """Analyze a company's potential to be the next big winner"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if len(hist) < 50:  # Need sufficient data
                return None
            
            # Basic company info
            company_name = info.get('longName', symbol)
            market_cap = info.get('marketCap', 0)
            
            # Skip if too large (mega caps) or too small (micro caps)
            if market_cap > 1_000_000_000_000 or market_cap < 100_000_000:  # $100M - $1T range
                return None
            
            # Financial metrics
            revenue_growth_yoy = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            revenue_growth_qoq = info.get('quarterlyRevenueGrowth', 0) * 100 if info.get('quarterlyRevenueGrowth') else 0
            gross_margin = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
            
            # Innovation indicators
            r_and_d_intensity = self._calculate_rd_intensity(info)
            
            # Ownership structure
            insider_ownership = info.get('heldByInsiders', 0) * 100 if info.get('heldByInsiders') else 0
            institutional_ownership = info.get('heldByInstitutions', 0) * 100 if info.get('heldByInstitutions') else 0
            short_interest = info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0
            
            # Technical analysis
            price_momentum_score = self._calculate_price_momentum(hist)
            volume_accumulation_score = self._calculate_volume_accumulation(hist)
            
            # Fundamental scores
            innovation_score = self._calculate_innovation_score(info, theme)
            market_opportunity_score = self._calculate_market_opportunity_score(info, theme)
            competitive_moat_score = self._calculate_competitive_moat_score(info)
            execution_score = self._calculate_execution_score(info)
            
            # Calculate total winner score
            total_winner_score = self._calculate_total_winner_score(
                revenue_growth_yoy, revenue_growth_qoq, gross_margin, r_and_d_intensity,
                price_momentum_score, volume_accumulation_score, innovation_score,
                market_opportunity_score, competitive_moat_score, execution_score
            )
            
            # Risk assessment
            risk_level = self._assess_risk_level(market_cap, revenue_growth_yoy, short_interest)
            
            # Catalyst potential
            catalyst_potential = self._assess_catalyst_potential(info, theme)
            
            return WinnerCandidate(
                symbol=symbol,
                company_name=company_name,
                disruptive_theme=theme,
                market_cap=market_cap,
                revenue_growth_yoy=revenue_growth_yoy,
                revenue_growth_qoq=revenue_growth_qoq,
                gross_margin=gross_margin,
                r_and_d_intensity=r_and_d_intensity,
                insider_ownership=insider_ownership,
                institutional_ownership=institutional_ownership,
                short_interest=short_interest,
                price_momentum_score=price_momentum_score,
                volume_accumulation_score=volume_accumulation_score,
                innovation_score=innovation_score,
                market_opportunity_score=market_opportunity_score,
                competitive_moat_score=competitive_moat_score,
                execution_score=execution_score,
                total_winner_score=total_winner_score,
                risk_level=risk_level,
                catalyst_potential=catalyst_potential,
                scan_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Winner analysis failed for {symbol}: {e}", "WINNER_SCANNER")
            return None
    
    def _calculate_rd_intensity(self, info: Dict) -> float:
        """Calculate R&D intensity as % of revenue"""
        try:
            total_revenue = info.get('totalRevenue', 0)
            # R&D not directly available, estimate from operating expenses
            operating_expenses = info.get('totalOperatingExpenses', 0)
            if total_revenue > 0 and operating_expenses > 0:
                # Estimate R&D as portion of operating expenses (varies by industry)
                estimated_rd = operating_expenses * 0.15  # Conservative estimate
                return (estimated_rd / total_revenue) * 100
            return 0
        except:
            return 0
    
    def _calculate_price_momentum(self, hist: pd.DataFrame) -> float:
        """Calculate price momentum score (0-100)"""
        try:
            close = hist['Close']
            
            # Multiple timeframe momentum
            momentum_1m = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0
            momentum_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else 0
            momentum_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else 0
            momentum_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
            
            # Weight recent momentum more heavily
            weighted_momentum = (momentum_1m * 0.4 + momentum_3m * 0.3 + 
                               momentum_6m * 0.2 + momentum_1y * 0.1)
            
            # Convert to 0-100 score
            if weighted_momentum > 100:
                return 100
            elif weighted_momentum > 50:
                return 80 + (weighted_momentum - 50) * 0.4
            elif weighted_momentum > 0:
                return 50 + weighted_momentum
            elif weighted_momentum > -25:
                return 25 + (weighted_momentum + 25) * 1
            else:
                return 0
                
        except:
            return 50
    
    def _calculate_volume_accumulation(self, hist: pd.DataFrame) -> float:
        """Calculate volume accumulation score (0-100)"""
        try:
            volume = hist['Volume']
            close = hist['Close']
            
            # Volume trend analysis
            recent_volume = volume.tail(20).mean()
            older_volume = volume.head(20).mean()
            volume_trend = (recent_volume / older_volume - 1) * 100 if older_volume > 0 else 0
            
            # Price-volume correlation
            price_change = close.pct_change()
            volume_change = volume.pct_change()
            correlation = price_change.corr(volume_change)
            
            # On-balance volume trend
            obv = (volume * np.sign(close.diff())).cumsum()
            obv_trend = (obv.iloc[-1] / obv.iloc[0] - 1) * 100 if obv.iloc[0] != 0 else 0
            
            # Combine metrics
            score = 50  # Base score
            
            if volume_trend > 20:
                score += 25
            elif volume_trend > 0:
                score += 10
            
            if correlation > 0.3:
                score += 15
            elif correlation > 0:
                score += 5
            
            if obv_trend > 0:
                score += 10
            
            return min(100, max(0, score))
            
        except:
            return 50
    
    def _calculate_innovation_score(self, info: Dict, theme: DisruptiveTheme) -> float:
        """Calculate innovation score based on company characteristics"""
        score = 50  # Base score
        
        # Industry-specific innovation indicators
        if theme in [DisruptiveTheme.AI_ML, DisruptiveTheme.QUANTUM_COMPUTING]:
            # High-tech innovation
            score += 20
        elif theme in [DisruptiveTheme.BIOTECH, DisruptiveTheme.SPACE_TECH]:
            # Research-intensive innovation
            score += 25
        elif theme in [DisruptiveTheme.FINTECH, DisruptiveTheme.CYBERSECURITY]:
            # Software innovation
            score += 15
        
        # Company age (younger = more innovative potential)
        try:
            founded_year = info.get('founded', 2000)
            if founded_year > 2015:
                score += 20  # Very young company
            elif founded_year > 2010:
                score += 15  # Young company
            elif founded_year > 2000:
                score += 10  # Mature but not old
        except:
            pass
        
        # Employee growth (proxy for innovation)
        employees = info.get('fullTimeEmployees', 0)
        if 100 <= employees <= 5000:  # Sweet spot for innovation
            score += 15
        elif employees <= 10000:
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_market_opportunity_score(self, info: Dict, theme: DisruptiveTheme) -> float:
        """Calculate total addressable market opportunity score"""
        score = 50  # Base score
        
        # Theme-based market size scoring
        mega_markets = [DisruptiveTheme.AI_ML, DisruptiveTheme.CLOUD_INFRASTRUCTURE, 
                       DisruptiveTheme.CYBERSECURITY, DisruptiveTheme.FINTECH]
        
        large_markets = [DisruptiveTheme.BIOTECH, DisruptiveTheme.CLEAN_ENERGY,
                        DisruptiveTheme.AUTONOMOUS_VEHICLES]
        
        emerging_markets = [DisruptiveTheme.QUANTUM_COMPUTING, DisruptiveTheme.SPACE_TECH,
                           DisruptiveTheme.ROBOTICS]
        
        if theme in mega_markets:
            score += 30  # Trillion-dollar markets
        elif theme in large_markets:
            score += 25  # Hundred-billion markets
        elif theme in emerging_markets:
            score += 20  # Emerging but huge potential
        else:
            score += 15  # Other markets
        
        # Geographic reach
        try:
            country = info.get('country', 'US')
            if country == 'US':
                score += 15  # US market access
            else:
                score += 10  # International
        except:
            pass
        
        return min(100, max(0, score))
    
    def _calculate_competitive_moat_score(self, info: Dict) -> float:
        """Calculate competitive moat strength"""
        score = 50  # Base score
        
        # Gross margin as moat indicator
        gross_margin = info.get('grossMargins', 0)
        if gross_margin > 0.8:  # 80%+ margins
            score += 25
        elif gross_margin > 0.6:  # 60%+ margins
            score += 20
        elif gross_margin > 0.4:  # 40%+ margins
            score += 15
        elif gross_margin > 0.2:  # 20%+ margins
            score += 10
        
        # Return on equity
        roe = info.get('returnOnEquity', 0)
        if roe > 0.3:  # 30%+ ROE
            score += 15
        elif roe > 0.15:  # 15%+ ROE
            score += 10
        elif roe > 0:
            score += 5
        
        # Asset turnover (efficiency)
        total_revenue = info.get('totalRevenue', 0)
        total_assets = info.get('totalAssets', 1)
        asset_turnover = total_revenue / total_assets if total_assets > 0 else 0
        
        if asset_turnover > 1.5:
            score += 10
        elif asset_turnover > 1:
            score += 5
        
        return min(100, max(0, score))
    
    def _calculate_execution_score(self, info: Dict) -> float:
        """Calculate management execution score"""
        score = 50  # Base score
        
        # Revenue growth consistency
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.5:  # 50%+ growth
            score += 25
        elif revenue_growth > 0.3:  # 30%+ growth
            score += 20
        elif revenue_growth > 0.15:  # 15%+ growth
            score += 15
        elif revenue_growth > 0:
            score += 10
        
        # Profitability trend
        profit_margin = info.get('profitMargins', 0)
        if profit_margin > 0.2:  # 20%+ margins
            score += 15
        elif profit_margin > 0.1:  # 10%+ margins
            score += 10
        elif profit_margin > 0:  # Profitable
            score += 5
        
        # Cash position
        cash_per_share = info.get('totalCashPerShare', 0)
        if cash_per_share > 10:
            score += 10
        elif cash_per_share > 5:
            score += 5
        
        return min(100, max(0, score))
    
    def _calculate_total_winner_score(self, revenue_growth_yoy: float, revenue_growth_qoq: float,
                                    gross_margin: float, r_and_d_intensity: float,
                                    price_momentum_score: float, volume_accumulation_score: float,
                                    innovation_score: float, market_opportunity_score: float,
                                    competitive_moat_score: float, execution_score: float) -> float:
        """Calculate total winner score (0-100)"""
        
        # Growth component (30%)
        growth_score = min(100, (revenue_growth_yoy + revenue_growth_qoq) / 2)
        
        # Technical component (20%)
        technical_score = (price_momentum_score + volume_accumulation_score) / 2
        
        # Fundamental component (50%)
        fundamental_score = (innovation_score * 0.3 + market_opportunity_score * 0.25 +
                           competitive_moat_score * 0.25 + execution_score * 0.2)
        
        # Weighted total
        total_score = (growth_score * 0.3 + technical_score * 0.2 + fundamental_score * 0.5)
        
        # Bonus for exceptional metrics
        if revenue_growth_yoy > 100:  # 100%+ growth
            total_score += 10
        if gross_margin > 80:  # 80%+ margins
            total_score += 5
        if r_and_d_intensity > 20:  # 20%+ R&D intensity
            total_score += 5
        
        return min(100, max(0, total_score))
    
    def _assess_risk_level(self, market_cap: float, revenue_growth: float, short_interest: float) -> str:
        """Assess risk level of the candidate"""
        risk_factors = 0
        
        # Market cap risk
        if market_cap < 2_000_000_000:  # Under $2B
            risk_factors += 2
        elif market_cap < 10_000_000_000:  # Under $10B
            risk_factors += 1
        
        # Growth sustainability risk
        if revenue_growth > 200:  # Over 200% growth (unsustainable)
            risk_factors += 2
        elif revenue_growth < 20:  # Under 20% growth
            risk_factors += 1
        
        # Short interest risk
        if short_interest > 20:  # High short interest
            risk_factors += 2
        elif short_interest > 10:
            risk_factors += 1
        
        if risk_factors >= 4:
            return "VERY_HIGH"
        elif risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _assess_catalyst_potential(self, info: Dict, theme: DisruptiveTheme) -> str:
        """Assess potential catalysts for explosive growth"""
        catalysts = []
        
        # Theme-specific catalysts
        if theme == DisruptiveTheme.AI_ML:
            catalysts.append("AI adoption acceleration")
        elif theme == DisruptiveTheme.CYBERSECURITY:
            catalysts.append("Increasing cyber threats")
        elif theme == DisruptiveTheme.BIOTECH:
            catalysts.append("Drug approval pipeline")
        elif theme == DisruptiveTheme.CLEAN_ENERGY:
            catalysts.append("Green energy transition")
        
        # Company-specific catalysts
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.5:
            catalysts.append("Hypergrowth momentum")
        
        market_cap = info.get('marketCap', 0)
        if market_cap < 5_000_000_000:
            catalysts.append("Small cap expansion potential")
        
        if len(catalysts) >= 3:
            return "VERY_HIGH"
        elif len(catalysts) >= 2:
            return "HIGH"
        elif len(catalysts) >= 1:
            return "MODERATE"
        else:
            return "LOW"
    
    def scan_for_next_winners(self, min_score: float = 50) -> List[WinnerCandidate]:
        """Scan for next big winner candidates"""
        print("🚀 SCANNING FOR NEXT BIG WINNERS")
        print("=" * 50)
        print("🎯 Target: Next PLTR, CRWD, NVDA")
        print(f"📊 Minimum Score Threshold: {min_score}")
        print("=" * 50)
        
        all_candidates = []
        
        for theme, symbols in self.disruptive_universe.items():
            print(f"\n🔍 Scanning {theme.value}...")
            theme_candidates = []
            
            for symbol in symbols:
                try:
                    candidate = self.analyze_winner_potential(symbol, theme)
                    if candidate:
                        if candidate.total_winner_score >= min_score:
                            theme_candidates.append(candidate)
                            print(f"   ✅ {symbol}: {candidate.total_winner_score:.1f} score (${candidate.market_cap/1e9:.1f}B)")
                        else:
                            print(f"   📊 {symbol}: {candidate.total_winner_score:.1f} score (${candidate.market_cap/1e9:.1f}B) - below {min_score}")
                    else:
                        # Get basic info to see why it was filtered
                        try:
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            market_cap = info.get('marketCap', 0)
                            if market_cap == 0:
                                print(f"   ❌ {symbol}: No market cap data")
                            elif market_cap > 1_000_000_000_000:
                                print(f"   ❌ {symbol}: Too large (${market_cap/1e9:.1f}B)")
                            elif market_cap < 100_000_000:
                                print(f"   ❌ {symbol}: Too small (${market_cap/1e6:.1f}M)")
                            else:
                                print(f"   ❌ {symbol}: Insufficient data")
                        except:
                            print(f"   ❌ {symbol}: Data fetch failed")
                        
                except Exception as e:
                    print(f"   💥 {symbol}: Analysis failed - {e}")
                    continue
            
            all_candidates.extend(theme_candidates)
            print(f"   📊 {len(theme_candidates)} candidates found in {theme.value}")
        
        # Sort by total winner score
        all_candidates.sort(key=lambda x: x.total_winner_score, reverse=True)
        
        print(f"\n🏆 TOTAL CANDIDATES FOUND: {len(all_candidates)}")
        return all_candidates
    
    def display_winner_candidates(self, candidates: List[WinnerCandidate]):
        """Display winner candidates in a professional format"""
        if not candidates:
            print("\n❌ No next big winner candidates found")
            return
        
        print(f"\n🏆 NEXT BIG WINNER CANDIDATES")
        print("=" * 100)
        
        # Top 20 candidates
        top_candidates = candidates[:20]
        
        print(f"{'Rank':<4} {'Symbol':<8} {'Company':<25} {'Theme':<20} {'Score':<6} {'Growth':<8} {'Risk':<8}")
        print("-" * 100)
        
        for i, candidate in enumerate(top_candidates, 1):
            theme_short = candidate.disruptive_theme.value[:18] + ".." if len(candidate.disruptive_theme.value) > 20 else candidate.disruptive_theme.value
            company_short = candidate.company_name[:23] + ".." if len(candidate.company_name) > 25 else candidate.company_name
            
            print(f"{i:<4} {candidate.symbol:<8} {company_short:<25} {theme_short:<20} "
                  f"{candidate.total_winner_score:<6.1f} {candidate.revenue_growth_yoy:<8.1f} {candidate.risk_level:<8}")
        
        # Detailed analysis of top 5
        print(f"\n📊 DETAILED ANALYSIS - TOP 5 CANDIDATES")
        print("=" * 80)
        
        for i, candidate in enumerate(top_candidates[:5], 1):
            print(f"\n🥇 #{i} {candidate.symbol} - {candidate.company_name}")
            print(f"   🎯 Theme: {candidate.disruptive_theme.value}")
            print(f"   💰 Market Cap: ${candidate.market_cap/1e9:.1f}B")
            print(f"   📈 Revenue Growth: {candidate.revenue_growth_yoy:.1f}% YoY, {candidate.revenue_growth_qoq:.1f}% QoQ")
            print(f"   🏭 Gross Margin: {candidate.gross_margin:.1f}%")
            print(f"   🔬 R&D Intensity: {candidate.r_and_d_intensity:.1f}%")
            print(f"   📊 Scores: Innovation {candidate.innovation_score:.1f}, Market {candidate.market_opportunity_score:.1f}, Moat {candidate.competitive_moat_score:.1f}")
            print(f"   ⚠️  Risk Level: {candidate.risk_level}")
            print(f"   🚀 Catalyst Potential: {candidate.catalyst_potential}")
            print(f"   🏆 Total Winner Score: {candidate.total_winner_score:.1f}/100")
        
        # Theme breakdown
        theme_breakdown = {}
        for candidate in candidates:
            theme = candidate.disruptive_theme.value
            if theme not in theme_breakdown:
                theme_breakdown[theme] = []
            theme_breakdown[theme].append(candidate)
        
        print(f"\n📈 THEME BREAKDOWN")
        print("-" * 50)
        for theme, theme_candidates in sorted(theme_breakdown.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score = sum(c.total_winner_score for c in theme_candidates) / len(theme_candidates)
            print(f"   {theme:<30}: {len(theme_candidates):2d} candidates (avg score: {avg_score:.1f})")

def main():
    """Main scanner execution"""
    scanner = NextWinnerScanner()
    
    print("🎯 NEXT BIG WINNER SCANNER")
    print("Finding the next PLTR, CRWD, NVDA before they explode")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Scan for candidates with lower threshold first
    print("\n🔍 PHASE 1: Scanning with threshold 50...")
    candidates = scanner.scan_for_next_winners(min_score=50)
    
    # If we find candidates, also show higher quality ones
    if candidates:
        print("\n🔍 PHASE 2: High-quality candidates (60+ score)...")
        high_quality = [c for c in candidates if c.total_winner_score >= 60]
        print(f"Found {len(high_quality)} high-quality candidates")
        
        print("\n🔍 PHASE 3: Premium candidates (70+ score)...")
        premium = [c for c in candidates if c.total_winner_score >= 70]
        print(f"Found {len(premium)} premium candidates")
    
    # Display results
    scanner.display_winner_candidates(candidates)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Scan completed in {duration:.1f} seconds")
    print(f"🎯 {len(candidates)} next big winner candidates identified")
    
    if candidates:
        top_candidate = candidates[0]
        print(f"\n🏆 TOP PICK: {top_candidate.symbol} ({top_candidate.total_winner_score:.1f} score)")
        print(f"💡 {top_candidate.disruptive_theme.value}")
        print(f"🚀 Ready for explosive growth!")

if __name__ == "__main__":
    main()
