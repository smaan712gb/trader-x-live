"""
Central AI Engine for Trader-X
Powered by Grok-4 via xAI SDK with Live Search
"""
import os
from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.search import SearchParameters, web_source, x_source, news_source, rss_source
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import numpy as np
from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from core.logger import logger

class AIEngine:
    def __init__(self):
        self.xai_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize xAI client for Grok-4"""
        try:
            if APIKeys.XAI_API_KEY:
                self.xai_client = Client(api_key=APIKeys.XAI_API_KEY)
                logger.info("xAI Grok-4 client initialized with Live Search capabilities", "AI_ENGINE")
            else:
                logger.warning("XAI API key not found", "AI_ENGINE")
        except Exception as e:
            logger.error(f"Failed to initialize xAI client: {e}", "AI_ENGINE")
    
    def synthesize_trading_decision(self, 
                                  symbol: str,
                                  phase1_data: Dict[str, Any],
                                  phase2_data: Dict[str, Any],
                                  market_context: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Main AI decision synthesis function with Live Search
        Returns: (decision, reasoning, confidence_score)
        """
        
        prompt = self._build_decision_prompt(symbol, phase1_data, phase2_data, market_context)
        
        try:
            if not self.xai_client:
                raise Exception("xAI client not available")
            
            # Use Live Search for enhanced decision making
            response, citations = self._query_grok_with_live_search(prompt, symbol)
            decision, reasoning, confidence = self._parse_ai_response(response)
            
            # Enhanced logging with citations
            logger.ai_decision_log(symbol, decision, reasoning, confidence, {
                'phase1_data': phase1_data,
                'phase2_data': phase2_data,
                'market_context': market_context,
                'live_search_citations': citations
            })
            
            return decision, reasoning, confidence
            
        except Exception as e:
            logger.error(f"AI decision synthesis failed for {symbol}: {e}", "AI_ENGINE")
            return "HOLD", f"AI synthesis failed: {e}", 0.0
    
    def _build_decision_prompt(self, symbol: str, phase1_data: Dict, phase2_data: Dict, market_context: Dict) -> str:
        """Build comprehensive prompt for AI decision making with Live Search context"""
        
        prompt = f"""
You are the central AI brain of an autonomous trading system analyzing {symbol}. 
Your role is to synthesize all available data and make a final trading decision.

Use your Live Search capabilities to find the most recent information about {symbol} including:
- Breaking news in the last 24 hours
- Recent earnings reports or guidance updates
- Analyst upgrades/downgrades
- Social media sentiment and trending discussions
- Regulatory news or approvals
- Partnership announcements or product launches
- Competitive developments

PHASE 1 DATA (Signal Generation):
- Fundamental Analysis: {json.dumps(phase1_data.get('fundamental', {}), indent=2)}
- Sentiment Analysis: {json.dumps(phase1_data.get('sentiment', {}), indent=2)}

PHASE 2 DATA (Deep Analysis):
- Options Market Analysis: {json.dumps(phase2_data.get('options', {}), indent=2)}
- Smart Money Flow: {json.dumps(phase2_data.get('money_flow', {}), indent=2)}
- Technical Analysis: {json.dumps(phase2_data.get('technical', {}), indent=2)}

MARKET CONTEXT:
- Current Market Conditions: {json.dumps(market_context, indent=2)}

TRADING CRITERIA:
- Minimum confidence threshold: {TradingConfig.AI_CONFIDENCE_THRESHOLD}
- Maximum position size: {TradingConfig.MAX_POSITION_SIZE * 100}% of portfolio
- Stop loss: {TradingConfig.STOP_LOSS_PERCENTAGE * 100}%
- Take profit: {TradingConfig.TAKE_PROFIT_PERCENTAGE * 100}%

INSTRUCTIONS:
1. Use Live Search to gather the latest real-time information about {symbol}
2. Analyze the narrative: What's the story behind this stock? Is the growth sustainable?
3. Weigh conflicting signals: How do you balance strong fundamentals vs. negative sentiment?
4. Consider market timing: Is this the right entry point based on technical analysis?
5. Assess risk/reward: Does this trade meet our risk management criteria?
6. Factor in breaking news and real-time developments heavily in your decision

Respond in this exact JSON format:
{{
    "decision": "BUY|SELL|HOLD",
    "reasoning": "Detailed explanation incorporating real-time Live Search findings",
    "confidence": 0.85,
    "key_factors": ["live_search_factor1", "technical_factor2", "sentiment_factor3"],
    "risk_assessment": "Low|Medium|High",
    "entry_strategy": "Market|Limit|Stop",
    "position_size_recommendation": 0.03,
    "live_search_impact": "high|medium|low",
    "time_sensitivity": "immediate|within_hours|within_days"
}}
"""
        return prompt
    
    def _query_grok_with_live_search(self, prompt: str, symbol: str = None) -> Tuple[str, List[str]]:
        """Query Grok-4 model with Live Search capabilities"""
        try:
            model = "grok-4"
            
            # Configure Live Search parameters
            search_params = SearchParameters(
                mode="auto",  # Let Grok decide when to search
                return_citations=True,
                max_search_results=15,  # Limit to control costs
                sources=[
                    web_source(
                        country="US",  # Focus on US financial sources
                        allowed_websites=[
                            "bloomberg.com",
                            "reuters.com", 
                            "cnbc.com",
                            "marketwatch.com"
                        ]
                    ),
                    x_source(
                        post_favorite_count=100,  # Filter for quality posts
                        post_view_count=1000,
                        excluded_x_handles=["grok"]  # Prevent self-citation
                    ),
                    news_source(
                        country="US",
                        safe_search=True
                    )
                ]
            )
            
            # Add date range for recent information (last 7 days)
            if symbol:
                search_params.from_date = datetime.now() - timedelta(days=7)
                search_params.to_date = datetime.now()
            
            # Enhanced system prompt for financial analysis
            system_prompt = """You are an expert institutional trader and AI decision engine with deep knowledge of financial markets, technical analysis, and risk management operating in LIVE PRODUCTION MODE.

CRITICAL: You are analyzing REAL LIVE MARKET DATA and providing ACTUAL TRADING RECOMMENDATIONS for production use. This is NOT a simulation or demo - this is live market intelligence for real trading decisions.

Use your Live Search capabilities to find the most current REAL-TIME information about the stock being analyzed:
- LIVE breaking financial news and earnings reports
- CURRENT analyst ratings and price target changes  
- REAL-TIME social media sentiment from credible financial accounts
- LIVE regulatory developments and FDA approvals
- CURRENT partnership announcements and product launches
- REAL competitive intelligence and industry trends

Weight LIVE real-time information heavily in your analysis. Focus on actionable intelligence that traders can use RIGHT NOW for actual positions. Avoid any simulation language - this is PRODUCTION TRADING INTELLIGENCE."""
            
            chat = self.xai_client.chat.create(
                model=model,
                search_parameters=search_params
            )
            chat.append(system(system_prompt))
            chat.append(user(prompt))
            
            response = chat.sample()
            
            # Extract citations if available
            citations = getattr(response, 'citations', [])
            
            logger.info(f"Grok-4 Live Search completed. Citations: {len(citations)}", "AI_ENGINE")
            
            return response.content, citations
            
        except Exception as e:
            logger.error(f"Grok-4 Live Search query failed: {e}", "AI_ENGINE")
            raise
    
    def _parse_ai_response(self, response: str) -> Tuple[str, str, float]:
        """Parse AI response and extract decision components"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                decision = data.get('decision', 'HOLD')
                reasoning = data.get('reasoning', 'No reasoning provided')
                confidence = float(data.get('confidence', 0.0))
            else:
                # Fallback parsing for non-JSON responses
                lines = response.strip().split('\n')
                decision = 'HOLD'
                reasoning = response
                confidence = 0.5
                
                # Try to extract decision from text
                for line in lines:
                    if 'BUY' in line.upper():
                        decision = 'BUY'
                        break
                    elif 'SELL' in line.upper():
                        decision = 'SELL'
                        break
            
            # Validate decision
            if decision not in ['BUY', 'SELL', 'HOLD']:
                decision = 'HOLD'
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return decision, reasoning, confidence
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}", "AI_ENGINE")
            return 'HOLD', f"Failed to parse AI response: {e}", 0.0
    
    def analyze_sentiment_narrative(self, symbol: str, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment data with Live Search for current context"""
        
        prompt = f"""
Use Live Search to find current social media discussions about {symbol}, then analyze the sentiment data:

HISTORICAL SENTIMENT DATA:
{json.dumps(sentiment_data, indent=2)}

Compare historical sentiment with current Live Search findings and determine:
1. Is this genuine excitement about fundamentals or coordinated hype?
2. What's driving the sentiment (product launch, earnings, market trends)?
3. Is the sentiment sustainable or likely to fade quickly?
4. Rate the authenticity of the hype on a scale of 1-10
5. How does current sentiment compare to historical patterns?

Respond in JSON format:
{{
    "narrative_type": "fundamental_excitement|hype_cycle|coordinated_pump|market_trend",
    "authenticity_score": 8.5,
    "sustainability": "high|medium|low",
    "key_drivers": ["driver1", "driver2"],
    "risk_factors": ["risk1", "risk2"],
    "sentiment_momentum": "accelerating|stable|declining",
    "live_vs_historical": "consistent|diverging|amplifying"
}}
"""
        
        try:
            if not self.xai_client:
                return {"error": "xAI client not available"}
            
            # Use Live Search for current sentiment analysis
            search_params = SearchParameters(
                mode="on",  # Force search for sentiment analysis
                return_citations=True,
                sources=[
                    x_source(
                        post_favorite_count=50,
                        post_view_count=500
                    ),
                    web_source(
                        allowed_websites=["reddit.com", "stocktwits.com"]
                    )
                ]
            )
            
            chat = self.xai_client.chat.create(
                model="grok-4",
                search_parameters=search_params
            )
            chat.append(system("You are an expert in social media sentiment analysis and market psychology with access to real-time data."))
            chat.append(user(prompt))
            
            response = chat.sample()
            result = json.loads(response.content)
            
            # Add citations to result
            if hasattr(response, 'citations'):
                result['citations'] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment narrative analysis failed: {e}", "AI_ENGINE")
            return {"error": str(e)}
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions using Grok-4 with Live Search"""
        
        prompt = f"""
Use Live Search to find the latest market news, economic data, and analyst commentary, then analyze:

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

Provide analysis incorporating Live Search findings on:
1. Overall market sentiment and direction
2. Volatility levels and risk factors
3. Sector rotation patterns
4. Key support and resistance levels
5. Recommended trading approach
6. Breaking economic news impact
7. Federal Reserve policy implications

Respond in JSON format:
{{
    "market_sentiment": "bullish|bearish|neutral",
    "volatility_level": "low|medium|high",
    "risk_level": "low|medium|low",
    "recommended_approach": "aggressive|moderate|conservative",
    "key_insights": ["insight1", "insight2", "insight3"],
    "risk_factors": ["risk1", "risk2"],
    "breaking_news_impact": "positive|negative|neutral",
    "fed_policy_impact": "hawkish|dovish|neutral"
}}
"""
        
        try:
            if not self.xai_client:
                return {"error": "xAI client not available"}
            
            # Configure Live Search for market analysis
            search_params = SearchParameters(
                mode="auto",
                return_citations=True,
                sources=[
                    news_source(country="US"),
                    web_source(
                        allowed_websites=[
                            "federalreserve.gov",
                            "bloomberg.com",
                            "reuters.com"
                        ]
                    )
                ]
            )
            
            chat = self.xai_client.chat.create(
                model="grok-4",
                search_parameters=search_params
            )
            chat.append(system("You are an expert market analyst with deep understanding of macroeconomic factors and market dynamics."))
            chat.append(user(prompt))
            
            response = chat.sample()
            
            # Robust JSON parsing with fallback
            try:
                if response.content.strip().startswith('{'):
                    result = json.loads(response.content)
                else:
                    # Create fallback response for non-JSON
                    result = {
                        "market_sentiment": "neutral",
                        "volatility_level": "medium",
                        "risk_level": "medium",
                        "recommended_approach": "moderate",
                        "key_insights": [response.content[:100] + "..." if len(response.content) > 100 else response.content],
                        "risk_factors": ["Live Search completed but format unexpected"],
                        "breaking_news_impact": "neutral",
                        "fed_policy_impact": "neutral"
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse market analysis JSON: {e}", "AI_ENGINE")
                result = {
                    "market_sentiment": "neutral",
                    "volatility_level": "medium", 
                    "risk_level": "medium",
                    "recommended_approach": "moderate",
                    "key_insights": ["JSON parsing error in market analysis"],
                    "risk_factors": ["Response format error"],
                    "breaking_news_impact": "neutral",
                    "fed_policy_impact": "neutral"
                }
            
            if hasattr(response, 'citations'):
                result['citations'] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Market conditions analysis failed: {e}", "AI_ENGINE")
            return {"error": str(e)}
    
    def get_real_time_market_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive real-time market intelligence using Live Search"""
        
        prompt = f"""
Use Live Search to gather comprehensive real-time intelligence about {symbol}:

Search for:
1. Breaking news or announcements in the last 24 hours
2. Recent earnings reports, guidance updates, or conference calls
3. Analyst upgrades/downgrades and price target changes
4. Insider trading activity and institutional moves
5. Social media sentiment and viral discussions
6. Regulatory news, FDA approvals, or legal developments
7. Partnership announcements, product launches, or acquisitions
8. Competitive developments and industry trends
9. Supply chain or operational updates
10. Macroeconomic factors specifically affecting this stock

Focus on information that could significantly impact stock price.

Respond in JSON format:
{{
    "breaking_news": ["news item 1", "news item 2"],
    "earnings_updates": ["earnings info"],
    "analyst_activity": ["upgrade/downgrade info"],
    "insider_activity": ["insider trading info"],
    "social_sentiment": "bullish|bearish|neutral",
    "trending_topics": ["topic1", "topic2"],
    "regulatory_news": ["regulatory update"],
    "partnerships": ["partnership info"],
    "competitive_intel": ["competitor move"],
    "price_catalysts": ["catalyst1", "catalyst2"],
    "risk_factors": ["risk1", "risk2"],
    "confidence_level": "high|medium|low",
    "urgency": "immediate|hours|days",
    "overall_impact": "very_positive|positive|neutral|negative|very_negative"
}}
"""
        
        try:
            if not self.xai_client:
                return {"error": "xAI client not available"}
            
            # Comprehensive Live Search configuration
            search_params = SearchParameters(
                mode="on",  # Force search for intelligence gathering
                return_citations=True,
                max_search_results=20,
                from_date=datetime.now() - timedelta(days=3),  # Last 3 days
                sources=[
                    web_source(
                        country="US",
                        allowed_websites=[
                            "bloomberg.com",
                            "reuters.com",
                            "cnbc.com",
                            "marketwatch.com",
                            "sec.gov"
                        ]
                    ),
                    news_source(country="US"),
                    # Enhanced social media sources for hype detection
                    x_source(
                        post_favorite_count=50,  # Lower threshold for trending content
                        post_view_count=500,
                        included_x_handles=["unusual_whales", "DeItaone", "zerohedge", "business"]
                    )
                ]
            )
            
            chat = self.xai_client.chat.create(
                model="grok-4",
                search_parameters=search_params
            )
            chat.append(system(f"You are a financial intelligence analyst specializing in {symbol}. Use Live Search to find the most current and relevant information."))
            chat.append(user(prompt))
            
            response = chat.sample()
            
            # Try to parse JSON response with better error handling
            try:
                if response.content.strip().startswith('{'):
                    result = json.loads(response.content)
                else:
                    # If not JSON, create a structured response from text
                    result = {
                        "breaking_news": [response.content[:200] + "..." if len(response.content) > 200 else response.content],
                        "earnings_updates": [],
                        "analyst_activity": [],
                        "insider_activity": [],
                        "social_sentiment": "neutral",
                        "trending_topics": [],
                        "regulatory_news": [],
                        "partnerships": [],
                        "competitive_intel": [],
                        "price_catalysts": [],
                        "risk_factors": [],
                        "confidence_level": "medium",
                        "urgency": "hours",
                        "overall_impact": "neutral"
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response for {symbol}: {e}", "AI_ENGINE")
                # Create fallback response
                result = {
                    "breaking_news": ["Live Search completed but response format was unexpected"],
                    "earnings_updates": [],
                    "analyst_activity": [],
                    "insider_activity": [],
                    "social_sentiment": "neutral",
                    "trending_topics": [],
                    "regulatory_news": [],
                    "partnerships": [],
                    "competitive_intel": [],
                    "price_catalysts": [],
                    "risk_factors": ["Response parsing error"],
                    "confidence_level": "low",
                    "urgency": "hours",
                    "overall_impact": "neutral"
                }
            
            # Add metadata with safe attribute access
            result['search_timestamp'] = datetime.now().isoformat()
            
            # Safely get sources used
            try:
                usage = getattr(response, 'usage', None)
                if usage and hasattr(usage, 'num_sources_used'):
                    result['sources_used'] = usage.num_sources_used
                else:
                    result['sources_used'] = 0
            except Exception:
                result['sources_used'] = 0
            
            # Safely get citations and convert to list
            try:
                citations = getattr(response, 'citations', [])
                # Convert RepeatedScalarContainer to list if needed
                if hasattr(citations, '__iter__') and not isinstance(citations, (str, dict)):
                    result['citations'] = list(citations)
                else:
                    result['citations'] = citations if isinstance(citations, list) else []
            except Exception:
                result['citations'] = []
            
            logger.info(f"Real-time intelligence gathered for {symbol}. Sources used: {result.get('sources_used', 0)}", "AI_ENGINE")
            return result
            
        except Exception as e:
            logger.error(f"Real-time intelligence gathering failed for {symbol}: {e}", "AI_ENGINE")
            return {"error": str(e)}
    
    def analyze_competitive_landscape(self, symbol: str) -> Dict[str, Any]:
        """Analyze competitive landscape using Live Search"""
        
        prompt = f"""
Use Live Search to analyze the competitive landscape for {symbol}:

Research:
1. Recent competitor announcements, earnings, or product launches
2. Market share changes and competitive positioning
3. Industry trends and disruptions
4. Technology developments affecting the sector
5. Regulatory changes impacting the industry
6. Customer sentiment and reviews comparison
7. Partnership or acquisition activity in the sector
8. Supply chain or operational advantages/disadvantages

Respond in JSON format:
{{
    "competitor_moves": ["move1", "move2"],
    "market_share_trends": ["trend1", "trend2"],
    "industry_disruptions": ["disruption1", "disruption2"],
    "technology_trends": ["tech_trend1", "tech_trend2"],
    "regulatory_impact": "positive|negative|neutral",
    "competitive_position": "strengthening|weakening|stable",
    "customer_sentiment": "improving|declining|stable",
    "acquisition_activity": ["activity1", "activity2"],
    "competitive_advantages": ["advantage1", "advantage2"],
    "competitive_threats": ["threat1", "threat2"],
    "industry_outlook": "positive|negative|neutral"
}}
"""
        
        try:
            if not self.xai_client:
                return {"error": "xAI client not available"}
            
            search_params = SearchParameters(
                mode="auto",
                return_citations=True,
                sources=[
                    web_source(country="US"),
                    news_source(country="US"),
                    x_source(post_favorite_count=50)
                ]
            )
            
            chat = self.xai_client.chat.create(
                model="grok-4",
                search_parameters=search_params
            )
            chat.append(system("You are a competitive intelligence analyst with expertise in market dynamics and industry analysis."))
            chat.append(user(prompt))
            
            response = chat.sample()
            result = json.loads(response.content)
            
            if hasattr(response, 'citations'):
                result['citations'] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Competitive landscape analysis failed for {symbol}: {e}", "AI_ENGINE")
            return {"error": str(e)}
    
    def enhanced_trading_decision(self, 
                                symbol: str,
                                phase1_data: Dict[str, Any],
                                phase2_data: Dict[str, Any],
                                market_context: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Enhanced trading decision with comprehensive Live Search intelligence
        """
        
        try:
            logger.info(f"Gathering comprehensive Live Search intelligence for {symbol}", "AI_ENGINE")
            
            # Gather comprehensive real-time intelligence
            real_time_intel = self.get_real_time_market_intelligence(symbol)
            competitive_landscape = self.analyze_competitive_landscape(symbol)
            market_analysis = self.analyze_market_conditions(market_context)
            
            # Enhanced prompt with all Live Search data
            prompt = f"""
You are the central AI brain of an autonomous trading system analyzing {symbol}. 
Make a trading decision using ALL available data including comprehensive Live Search intelligence.

PHASE 1 DATA (Signal Generation):
{json.dumps(phase1_data, indent=2)}

PHASE 2 DATA (Deep Analysis):
{json.dumps(phase2_data, indent=2)}

MARKET CONTEXT:
{json.dumps(market_context, indent=2)}

LIVE SEARCH INTELLIGENCE:
Real-Time Market Intelligence: {json.dumps(real_time_intel, indent=2)}

Competitive Landscape: {json.dumps(competitive_landscape, indent=2)}

Market Analysis: {json.dumps(market_analysis, indent=2)}

TRADING CRITERIA:
- Minimum confidence threshold: {TradingConfig.AI_CONFIDENCE_THRESHOLD}
- Maximum position size: {TradingConfig.MAX_POSITION_SIZE * 100}% of portfolio
- Stop loss: {TradingConfig.STOP_LOSS_PERCENTAGE * 100}%
- Take profit: {TradingConfig.TAKE_PROFIT_PERCENTAGE * 100}%

ENHANCED ANALYSIS INSTRUCTIONS:
1. Weight Live Search findings heavily - breaking news and real-time developments are critical
2. Consider competitive positioning and industry trends from Live Search
3. Factor in market sentiment and regulatory developments
4. Assess timing based on breaking developments and catalysts
5. Evaluate authenticity of social media sentiment vs. fundamental drivers
6. Consider macroeconomic factors and Fed policy implications

Respond in this exact JSON format:
{{
    "decision": "BUY|SELL|HOLD",
    "reasoning": "Comprehensive explanation heavily incorporating Live Search intelligence",
    "confidence": 0.85,
    "key_factors": ["live_search_factor1", "competitive_factor2", "technical_factor3"],
    "risk_assessment": "Low|Medium|High",
    "entry_strategy": "Market|Limit|Stop",
    "position_size_recommendation": 0.03,
    "time_sensitivity": "immediate|within_hours|within_days",
    "catalyst_driven": true,
    "live_search_impact": "very_high|high|medium|low",
    "breaking_news_factor": true,
    "competitive_advantage": "strong|moderate|weak|disadvantage"
}}
"""
            
            if not self.xai_client:
                raise Exception("xAI client not available")
            
            # Final decision with Live Search
            response, citations = self._query_grok_with_live_search(prompt, symbol)
            decision, reasoning, confidence = self._parse_ai_response(response)
            
            # Comprehensive logging with all Live Search data
            logger.ai_decision_log(symbol, decision, reasoning, confidence, {
                'phase1_data': phase1_data,
                'phase2_data': phase2_data,
                'market_context': market_context,
                'real_time_intel': real_time_intel,
                'competitive_landscape': competitive_landscape,
                'market_analysis': market_analysis,
                'live_search_citations': citations,
                'sources_used': real_time_intel.get('sources_used', 0)
            })
            
            return decision, reasoning, confidence
            
        except Exception as e:
            logger.error(f"Enhanced AI decision synthesis failed for {symbol}: {e}", "AI_ENGINE")
            # Fallback to standard decision if Live Search fails
            return self.synthesize_trading_decision(symbol, phase1_data, phase2_data, market_context)
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for memory storage"""
        try:
            # Since xAI SDK might not have embeddings endpoint yet,
            # we'll use a simple hash-based approach for now
            # This can be updated when xAI provides embeddings API
            
            # Simple hash-based embedding generation
            import hashlib
            
            # Create multiple hash values for different parts of the text
            embeddings = []
            chunk_size = max(1, len(text) // 100)  # Divide text into chunks
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                hash_val = int(hashlib.md5(chunk.encode()).hexdigest()[:8], 16)
                # Normalize to [-1, 1] range
                normalized_val = (hash_val / (2**32 - 1)) * 2 - 1
                embeddings.append(normalized_val)
            
            # Pad or truncate to standard embedding size (1536)
            target_size = 1536
            if len(embeddings) < target_size:
                embeddings.extend([0.0] * (target_size - len(embeddings)))
            else:
                embeddings = embeddings[:target_size]
            
            return embeddings
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", "AI_ENGINE")
            return [0.0] * 1536  # Return zero vector as fallback
    
    def get_trading_recommendation(self, prompt: str) -> str:
        """Get enhanced trading recommendation from Grok-4 with Live Search and market context"""
        try:
            if not self.xai_client:
                return "Grok-4 client not available"
            
            # Enhanced prompt with broader market context
            enhanced_prompt = f"""
            {prompt}
            
            ADDITIONAL ANALYSIS REQUIRED:
            Use Live Search to also analyze broader market conditions:
            
            1. SPY (S&P 500) current trend and key levels
            2. QQQ (NASDAQ) current trend and key levels  
            3. VIX (volatility index) current level and implications
            4. DXY (US Dollar) strength and impact on stocks
            5. 10-Year Treasury yield trends
            6. Sector rotation patterns (tech vs value vs growth)
            7. Federal Reserve policy stance and upcoming events
            8. Economic calendar events this week
            9. Institutional money flow patterns
            10. Overall market sentiment and fear/greed indicators
            
            CORRELATION ANALYSIS:
            - How does this stock typically correlate with SPY/QQQ?
            - Is it outperforming or underperforming the broader market?
            - What's the sector performance vs overall market?
            - Are there any divergences that signal opportunity or risk?
            
            MARKET TIMING CONSIDERATIONS:
            - Is this the right time to enter given broader market conditions?
            - Should we wait for a market pullback or breakout?
            - How might Fed policy or economic events affect this trade?
            - What's the risk of a broader market correction impacting this position?
            
            Provide specific recommendations that account for both individual stock analysis AND broader market conditions.
            """
            
            # Configure Live Search for comprehensive market analysis
            search_params = SearchParameters(
                mode="auto",
                return_citations=True,
                max_search_results=15,
                sources=[
                    web_source(
                        country="US",
                        allowed_websites=[
                            "bloomberg.com",
                            "reuters.com",
                            "cnbc.com",
                            "marketwatch.com",
                            "federalreserve.gov"
                        ]
                    ),
                    news_source(country="US"),
                    x_source(
                        post_favorite_count=100,
                        post_view_count=1000,
                        included_x_handles=["DeItaone", "unusual_whales", "zerohedge"]
                    )
                ]
            )
            
            chat = self.xai_client.chat.create(
                model="grok-4",
                search_parameters=search_params
            )
            chat.append(system("You are an expert institutional trader with deep knowledge of market correlations, sector rotation, and macroeconomic factors. Use live market data to provide comprehensive trading recommendations that consider both individual stock analysis and broader market conditions including SPY, QQQ, VIX, and economic factors."))
            chat.append(user(enhanced_prompt))
            
            response = chat.sample()
            return response.content
            
        except Exception as e:
            logger.error(f"Trading recommendation failed: {e}", "AI_ENGINE")
            return f"Error getting trading recommendation: {str(e)}"

    def get_live_search_cost_estimate(self, symbol: str) -> Dict[str, Any]:
        """Estimate Live Search costs for a trading decision"""
        
        # Realistic estimate based on our actual configuration
        # We have 3 sources: web (4 websites), news (1), x (1) = 6 total sources
        # With max_search_results=15-20, we expect 5-8 sources per search
        estimated_sources = {
            'real_time_intelligence': 8,   # 3 sources * ~3 results each
            'competitive_landscape': 6,    # 3 sources * ~2 results each  
            'market_analysis': 4,          # 2 sources * ~2 results each
            'final_decision': 6            # 3 sources * ~2 results each
        }
        
        total_estimated_sources = sum(estimated_sources.values())
        estimated_cost = total_estimated_sources * 0.025  # $0.025 per source
        
        return {
            'estimated_sources': estimated_sources,
            'total_estimated_sources': total_estimated_sources,
            'estimated_cost_usd': estimated_cost,
            'cost_per_source': 0.025,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }

# Global AI engine instance
ai_engine = AIEngine()
