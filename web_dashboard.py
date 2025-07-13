"""
Trader-X Web Dashboard
Real-time visualization of Grok-4 Live Search trading intelligence
"""
import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from core.ai_engine import ai_engine
from technical_analysis_engine import technical_engine
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Trader-X Live Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Railway deployment configuration
import os
if os.getenv('RAILWAY_ENVIRONMENT_NAME'):
    # Running on Railway - optimize for production
    st.markdown("""
    <style>
        .stApp > header {visibility: hidden;}
        .stApp > footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .live-indicator {
        color: #00ff00;
        font-weight: bold;
    }
    .citation-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.8rem;
    }
    .decision-buy {
        color: #00aa00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .decision-sell {
        color: #aa0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .decision-hold {
        color: #aa6600;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Trader-X Live Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Powered by Grok-4 Live Search • Real-time Market Intelligence</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="NVDA", help="Enter a stock symbol (e.g., AAPL, TSLA, NVDA)")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Complete Intelligence", "Technical Analysis", "Real-time News", "Competitive Analysis", "Market Conditions", "Cost Estimation"]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now") or auto_refresh:
        st.rerun()
    
    # Live status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="live-indicator">🟢 LIVE</p>', unsafe_allow_html=True)
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content area
    if symbol:
        if analysis_type == "Complete Intelligence":
            show_complete_intelligence(symbol)
        elif analysis_type == "Technical Analysis":
            show_technical_analysis(symbol)
        elif analysis_type == "Real-time News":
            show_real_time_intelligence(symbol)
        elif analysis_type == "Competitive Analysis":
            show_competitive_analysis(symbol)
        elif analysis_type == "Market Conditions":
            show_market_conditions()
        elif analysis_type == "Cost Estimation":
            show_cost_estimation(symbol)
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def show_complete_intelligence(symbol):
    """Display complete Live Search intelligence analysis"""
    st.header(f"📊 Complete Intelligence Analysis: {symbol}")
    
    with st.spinner("🔍 Gathering comprehensive Live Search intelligence..."):
        try:
            # Get cost estimate first
            cost_estimate = ai_engine.get_live_search_cost_estimate(symbol)
            
            # Display cost estimate
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Estimated Sources", cost_estimate['total_estimated_sources'])
            with col2:
                st.metric("Estimated Cost", f"${cost_estimate['estimated_cost_usd']:.2f}")
            with col3:
                st.metric("Cost per Source", f"${cost_estimate['cost_per_source']:.3f}")
            with col4:
                st.metric("Analysis Time", "~2-3 min")
            
            # Get real-time intelligence
            real_time_intel = ai_engine.get_real_time_market_intelligence(symbol)
            
            if 'error' not in real_time_intel:
                # Display key metrics
                st.subheader("🎯 Key Intelligence Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sources Used", real_time_intel.get('sources_used', 0))
                with col2:
                    st.metric("Citations", len(real_time_intel.get('citations', [])))
                with col3:
                    sentiment_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
                    sentiment = real_time_intel.get('social_sentiment', 'neutral')
                    st.metric("Social Sentiment", f"{sentiment_color.get(sentiment, '🟡')} {sentiment.title()}")
                with col4:
                    impact_color = {"very_positive": "🟢", "positive": "🟢", "neutral": "🟡", "negative": "🔴", "very_negative": "🔴"}
                    impact = real_time_intel.get('overall_impact', 'neutral')
                    st.metric("Overall Impact", f"{impact_color.get(impact, '🟡')} {impact.replace('_', ' ').title()}")
                
                # Breaking News Section
                st.subheader("📰 Breaking News & Developments")
                breaking_news = real_time_intel.get('breaking_news', [])
                if breaking_news:
                    for i, news in enumerate(breaking_news[:5]):  # Show top 5
                        st.markdown(f"**{i+1}.** {news}")
                else:
                    st.info("No breaking news found in the last 24 hours")
                
                # Create tabs for different intelligence categories
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Intelligence", "🏢 Competitive Intel", "⚠️ Risk Factors", "🔗 Sources"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analyst Activity")
                        analyst_activity = real_time_intel.get('analyst_activity', [])
                        if analyst_activity:
                            for activity in analyst_activity:
                                st.markdown(f"• {activity}")
                        else:
                            st.info("No recent analyst activity")
                        
                        st.subheader("🎯 Price Catalysts")
                        catalysts = real_time_intel.get('price_catalysts', [])
                        if catalysts:
                            for catalyst in catalysts:
                                st.markdown(f"• {catalyst}")
                        else:
                            st.info("No immediate price catalysts identified")
                    
                    with col2:
                        st.subheader("📋 Earnings Updates")
                        earnings = real_time_intel.get('earnings_updates', [])
                        if earnings:
                            for earning in earnings:
                                st.markdown(f"• {earning}")
                        else:
                            st.info("No recent earnings updates")
                        
                        st.subheader("🤝 Partnerships & Deals")
                        partnerships = real_time_intel.get('partnerships', [])
                        if partnerships:
                            for partnership in partnerships:
                                st.markdown(f"• {partnership}")
                        else:
                            st.info("No recent partnership announcements")
                
                with tab2:
                    st.subheader("🏭 Competitive Intelligence")
                    competitive_intel = real_time_intel.get('competitive_intel', [])
                    if competitive_intel:
                        for intel in competitive_intel:
                            st.markdown(f"• {intel}")
                    else:
                        st.info("No significant competitive developments")
                    
                    st.subheader("📊 Trending Topics")
                    trending = real_time_intel.get('trending_topics', [])
                    if trending:
                        # Create a simple word cloud visualization
                        trending_df = pd.DataFrame({'Topic': trending, 'Mentions': [1] * len(trending)})
                        fig = px.bar(trending_df, x='Topic', y='Mentions', title="Trending Topics")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trending topics identified")
                
                with tab3:
                    st.subheader("⚠️ Risk Factors")
                    risk_factors = real_time_intel.get('risk_factors', [])
                    if risk_factors:
                        for risk in risk_factors:
                            st.warning(f"⚠️ {risk}")
                    else:
                        st.success("✅ No significant risk factors identified")
                    
                    # Risk assessment summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        confidence = real_time_intel.get('confidence_level', 'medium')
                        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                        st.metric("Confidence Level", f"{confidence_color.get(confidence, '🟡')} {confidence.title()}")
                    with col2:
                        urgency = real_time_intel.get('urgency', 'hours')
                        st.metric("Time Sensitivity", urgency.replace('_', ' ').title())
                    with col3:
                        timestamp = real_time_intel.get('search_timestamp', datetime.now().isoformat())
                        search_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        st.metric("Analysis Time", search_time.strftime('%H:%M:%S'))
                
                with tab4:
                    st.subheader("🔗 Live Search Citations")
                    citations = real_time_intel.get('citations', [])
                    if citations:
                        st.write(f"**{len(citations)} sources consulted:**")
                        for i, citation in enumerate(citations[:10]):  # Show top 10
                            st.markdown(f'<div class="citation-box">{i+1}. <a href="{citation}" target="_blank">{citation}</a></div>', unsafe_allow_html=True)
                        if len(citations) > 10:
                            st.info(f"... and {len(citations) - 10} more sources")
                    else:
                        st.info("No citations available")
            
            else:
                st.error(f"Error gathering intelligence: {real_time_intel['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_real_time_intelligence(symbol):
    """Display real-time market intelligence"""
    st.header(f"📡 Real-time Intelligence: {symbol}")
    
    with st.spinner("🔍 Gathering real-time market intelligence..."):
        try:
            intel = ai_engine.get_real_time_market_intelligence(symbol)
            
            if 'error' not in intel:
                # Key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", intel.get('sources_used', 0))
                with col2:
                    st.metric("Breaking News Items", len(intel.get('breaking_news', [])))
                with col3:
                    st.metric("Citations", len(intel.get('citations', [])))
                
                # Breaking news
                st.subheader("📰 Breaking News")
                for news in intel.get('breaking_news', []):
                    st.info(f"📰 {news}")
                
                # Social sentiment gauge
                sentiment = intel.get('social_sentiment', 'neutral')
                sentiment_score = {"bullish": 0.8, "neutral": 0.5, "bearish": 0.2}.get(sentiment, 0.5)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Social Sentiment"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.7], 'color': "gray"},
                            {'range': [0.7, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Error: {intel['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_competitive_analysis(symbol):
    """Display competitive landscape analysis"""
    st.header(f"🏢 Competitive Analysis: {symbol}")
    
    with st.spinner("🔍 Analyzing competitive landscape..."):
        try:
            analysis = ai_engine.analyze_competitive_landscape(symbol)
            
            if 'error' not in analysis:
                # Competitive position overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    position = analysis.get('competitive_position', 'stable')
                    position_color = {"strengthening": "🟢", "stable": "🟡", "weakening": "🔴"}
                    st.metric("Competitive Position", f"{position_color.get(position, '🟡')} {position.title()}")
                with col2:
                    outlook = analysis.get('industry_outlook', 'neutral')
                    outlook_color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
                    st.metric("Industry Outlook", f"{outlook_color.get(outlook, '🟡')} {outlook.title()}")
                with col3:
                    sentiment = analysis.get('customer_sentiment', 'stable')
                    sentiment_color = {"improving": "🟢", "stable": "🟡", "declining": "🔴"}
                    st.metric("Customer Sentiment", f"{sentiment_color.get(sentiment, '🟡')} {sentiment.title()}")
                
                # Competitive advantages vs threats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💪 Competitive Advantages")
                    advantages = analysis.get('competitive_advantages', [])
                    if advantages:
                        for advantage in advantages:
                            st.success(f"✅ {advantage}")
                    else:
                        st.info("No specific advantages identified")
                
                with col2:
                    st.subheader("⚠️ Competitive Threats")
                    threats = analysis.get('competitive_threats', [])
                    if threats:
                        for threat in threats:
                            st.warning(f"⚠️ {threat}")
                    else:
                        st.success("No immediate threats identified")
                
                # Industry trends and disruptions
                st.subheader("📈 Industry Trends & Disruptions")
                trends = analysis.get('technology_trends', [])
                disruptions = analysis.get('industry_disruptions', [])
                
                if trends or disruptions:
                    for trend in trends:
                        st.info(f"📈 **Trend:** {trend}")
                    for disruption in disruptions:
                        st.warning(f"💥 **Disruption:** {disruption}")
                else:
                    st.info("No significant trends or disruptions identified")
                    
            else:
                st.error(f"Error: {analysis['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_market_conditions():
    """Display overall market conditions analysis"""
    st.header("🌍 Market Conditions Analysis")
    
    with st.spinner("🔍 Analyzing market conditions..."):
        try:
            # Mock market data for demonstration
            market_data = {
                "spy_price": 450.25,
                "vix": 18.5,
                "dxy": 103.2,
                "yield_10y": 4.25
            }
            
            conditions = ai_engine.analyze_market_conditions(market_data)
            
            if 'error' not in conditions:
                # Market overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment = conditions.get('market_sentiment', 'neutral')
                    sentiment_color = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
                    st.metric("Market Sentiment", f"{sentiment_color.get(sentiment, '🟡')} {sentiment.title()}")
                
                with col2:
                    volatility = conditions.get('volatility_level', 'medium')
                    vol_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                    st.metric("Volatility Level", f"{vol_color.get(volatility, '🟡')} {volatility.title()}")
                
                with col3:
                    risk = conditions.get('risk_level', 'medium')
                    risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                    st.metric("Risk Level", f"{risk_color.get(risk, '🟡')} {risk.title()}")
                
                with col4:
                    approach = conditions.get('recommended_approach', 'moderate')
                    st.metric("Recommended Approach", approach.title())
                
                # Key insights
                st.subheader("🔍 Key Market Insights")
                insights = conditions.get('key_insights', [])
                for insight in insights:
                    st.info(f"💡 {insight}")
                
                # Risk factors
                st.subheader("⚠️ Market Risk Factors")
                risk_factors = conditions.get('risk_factors', [])
                for risk in risk_factors:
                    st.warning(f"⚠️ {risk}")
                    
            else:
                st.error(f"Error: {conditions['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_technical_analysis(symbol):
    """Display comprehensive technical analysis with Grok-4 enhanced recommendations and live chart"""
    st.header(f"📈 Technical Analysis: {symbol}")
    
    with st.spinner("📊 Analyzing multi-timeframe charts and calculating Grok-4 enhanced entry signals..."):
        try:
            # Get comprehensive technical analysis
            tech_analysis = technical_engine.get_comprehensive_analysis(symbol)
            
            if 'error' not in tech_analysis:
                # Current price and key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${tech_analysis['current_price']}")
                with col2:
                    overall_score = tech_analysis['technical_score']['overall_score']
                    score_color = "🟢" if overall_score >= 60 else "🔴" if overall_score <= 40 else "🟡"
                    st.metric("Technical Score", f"{score_color} {overall_score}/100")
                with col3:
                    recommendation = tech_analysis['recommendation']['action']
                    rec_color = "🟢" if "Buy" in recommendation else "🔴" if "Sell" in recommendation else "🟡"
                    st.metric("Recommendation", f"{rec_color} {recommendation}")
                with col4:
                    confidence = tech_analysis['recommendation']['confidence']
                    st.metric("Confidence", f"{confidence}%")
                
                # Live Interactive Chart with Grok-4 Signals
                st.subheader("📊 Live Chart with Grok-4 Trading Signals")
                
                # Chart timeframe controls
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.write("**Chart Settings:**")
                    
                    # Time period selection
                    time_period = st.selectbox(
                        "Time Period",
                        ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y"],
                        index=2,  # Default to 1M
                        help="Select the time period for the chart"
                    )
                    
                    # Interval selection based on time period
                    if time_period in ["1D"]:
                        interval_options = ["1m", "2m", "5m", "15m", "30m", "1h"]
                        default_interval = 3  # 15m
                    elif time_period in ["5D"]:
                        interval_options = ["5m", "15m", "30m", "1h", "4h"]
                        default_interval = 2  # 30m
                    elif time_period in ["1M", "3M"]:
                        interval_options = ["15m", "30m", "1h", "4h", "1d"]
                        default_interval = 4  # 1d
                    else:  # 6M, YTD, 1Y, 2Y, 5Y
                        interval_options = ["1h", "4h", "1d", "1wk", "1mo"]
                        default_interval = 2  # 1d
                    
                    chart_interval = st.selectbox(
                        "Chart Interval",
                        interval_options,
                        index=default_interval,
                        help="Select the candle interval for the chart"
                    )
                    
                    # Chart type selection
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Candlestick", "Line", "Area"],
                        index=0,
                        help="Select the chart visualization type"
                    )
                    
                    # Technical indicators toggle
                    show_volume = st.checkbox("Show Volume", value=True)
                    show_ma = st.checkbox("Show Moving Averages", value=True)
                    show_signals = st.checkbox("Show Grok-4 Signals", value=True)
                
                with col2:
                    show_live_trading_chart(symbol, tech_analysis, time_period, chart_interval, chart_type, show_volume, show_ma, show_signals)
                
                # Get Grok-4 Enhanced Recommendation
                st.subheader("🤖 Grok-4 Enhanced Trading Intelligence")
                
                with st.spinner("🔍 Getting Grok-4 enhanced recommendation with live market data..."):
                    grok_enhanced = technical_engine.get_grok_enhanced_recommendation(symbol, tech_analysis)
                    
                    if grok_enhanced.get('grok_enhanced'):
                        # Display Grok-4 enhanced recommendation
                        grok_rec = grok_enhanced['recommendation']
                        live_context = grok_enhanced['live_context']
                        
                        # Live context metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            sentiment = live_context['social_sentiment']
                            sentiment_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
                            st.metric("Live Sentiment", f"{sentiment_color.get(sentiment, '🟡')} {sentiment.title()}")
                        with col2:
                            st.metric("Breaking News", f"📰 {live_context['breaking_news_count']} items")
                        with col3:
                            impact = live_context['overall_impact']
                            impact_color = {"very_positive": "🟢", "positive": "🟢", "neutral": "🟡", "negative": "🔴", "very_negative": "🔴"}
                            st.metric("Market Impact", f"{impact_color.get(impact, '🟡')} {impact.replace('_', ' ').title()}")
                        with col4:
                            st.metric("Live Sources", f"🔍 {live_context['sources_used']} sources")
                        
                        # Enhanced recommendation display with better styling
                        st.markdown("### 🎯 Grok-4 Enhanced Trading Recommendation")
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                            <h4 style="color: white; margin-bottom: 1rem;">🚀 LIVE MARKET INTELLIGENCE ANALYSIS</h4>
                            <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px; 
                                       color: white; line-height: 1.6; font-size: 1.1rem;">
                                {grok_rec.replace('**', '<strong>').replace('**', '</strong>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        # Fallback if Grok-4 enhancement fails
                        if 'error' in grok_enhanced:
                            st.warning(f"⚠️ Grok-4 enhancement unavailable: {grok_enhanced['error']}")
                        
                        fallback_rec = grok_enhanced.get('fallback_recommendation', tech_analysis['recommendation'])
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                                   border-left: 4px solid #007bff; margin: 1rem 0;">
                            <h4 style="color: #007bff; margin-bottom: 1rem;">📊 Technical Analysis Recommendation</h4>
                            <div style="color: #333; line-height: 1.6;">
                                <strong>Action:</strong> {fallback_rec['action']}<br>
                                <strong>Confidence:</strong> {fallback_rec['confidence']}%<br>
                                <strong>Reasoning:</strong> {fallback_rec['reasoning']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Entry/Exit Signals Section
                st.subheader("🎯 Entry/Exit Signals for $5-10 Gains")
                
                entry_signals = tech_analysis['entry_signals']
                best_signal = entry_signals.get('best_signal')
                
                if best_signal:
                    # Best signal highlight
                    signal_type = best_signal['type']
                    signal_color = "🟢" if "Long" in signal_type else "🔴" if "Short" in signal_type else "🟡"
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                        <h4>{signal_color} BEST SIGNAL: {signal_type}</h4>
                        <p><strong>Entry Price:</strong> ${best_signal['entry_price']}</p>
                        <p><strong>Target Price:</strong> ${best_signal['target_price']}</p>
                        <p><strong>Stop Loss:</strong> ${best_signal['stop_loss']}</p>
                        <p><strong>Potential Gain:</strong> ${best_signal['potential_gain']}</p>
                        <p><strong>Risk/Reward:</strong> {best_signal['risk_reward_ratio']}:1</p>
                        <p><strong>Confidence:</strong> {best_signal['confidence']}%</p>
                        <p><strong>Timeframe:</strong> {best_signal['timeframe']}</p>
                        <p><strong>Reasoning:</strong> {best_signal['reasoning']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # All signals table
                    if len(entry_signals['signals']) > 1:
                        st.subheader("📋 All Entry Signals")
                        signals_df = pd.DataFrame(entry_signals['signals'])
                        st.dataframe(signals_df, use_container_width=True)
                else:
                    st.warning("⚠️ No clear entry signals for $5-10 gains at current levels")
                
                # Multi-timeframe Analysis
                st.subheader("⏰ Multi-Timeframe Analysis")
                
                timeframes = tech_analysis['timeframe_analysis']
                
                # Create tabs for each timeframe
                tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily", "🕐 4-Hour", "🕑 1-Hour", "🕒 15-Min"])
                
                with tab1:
                    display_timeframe_analysis(timeframes['daily'], "Daily")
                
                with tab2:
                    display_timeframe_analysis(timeframes['4hour'], "4-Hour")
                
                with tab3:
                    display_timeframe_analysis(timeframes['1hour'], "1-Hour")
                
                with tab4:
                    display_timeframe_analysis(timeframes['15min'], "15-Min")
                
                # Support/Resistance Levels
                st.subheader("📊 Support & Resistance Levels")
                
                sr_levels = tech_analysis['support_resistance']
                current_price = tech_analysis['current_price']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔴 Resistance Levels:**")
                    for level in sr_levels['resistance_levels']:
                        distance = ((level - current_price) / current_price) * 100
                        st.write(f"• ${level} (+{distance:.1f}%)")
                    
                    if sr_levels['nearest_resistance']:
                        st.success(f"🎯 **Next Target:** ${sr_levels['nearest_resistance']}")
                
                with col2:
                    st.write("**🟢 Support Levels:**")
                    for level in sr_levels['support_levels']:
                        distance = ((current_price - level) / current_price) * 100
                        st.write(f"• ${level} (-{distance:.1f}%)")
                    
                    if sr_levels['nearest_support']:
                        st.info(f"🛡️ **Key Support:** ${sr_levels['nearest_support']}")
                
                # High/Low Analysis
                st.subheader("📈 High/Low Analysis")
                
                high_low = tech_analysis['high_low_analysis']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Today's Range:**")
                    st.metric("High", f"${high_low['current_day']['high']}")
                    st.metric("Low", f"${high_low['current_day']['low']}")
                    st.metric("Position", f"{high_low['current_day']['position_pct']}%")
                
                with col2:
                    st.write("**Previous Day:**")
                    st.metric("High", f"${high_low['previous_day']['high']}")
                    st.metric("Low", f"${high_low['previous_day']['low']}")
                    st.metric("Range", f"${high_low['previous_day']['range']}")
                
                with col3:
                    st.write("**Week Range:**")
                    st.metric("High", f"${high_low['current_week']['high']}")
                    st.metric("Low", f"${high_low['current_week']['low']}")
                    st.metric("Position", f"{high_low['current_week']['position_pct']}%")
                
                # Technical Score Breakdown
                st.subheader("🎯 Technical Score Breakdown")
                
                tech_score = tech_analysis['technical_score']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{tech_score['overall_score']}/100")
                with col2:
                    st.metric("Recommendation", tech_score['recommendation'])
                with col3:
                    st.metric("Timeframe Alignment", f"{tech_score['timeframe_alignment']:.0f}%")
                
                # Final Recommendation
                st.subheader("🎯 Final Trading Recommendation")
                
                recommendation = tech_analysis['recommendation']
                
                rec_style = ""
                if "Buy" in recommendation['action']:
                    rec_style = "background-color: #d4edda; border-left: 4px solid #28a745; color: #155724;"
                elif "Sell" in recommendation['action']:
                    rec_style = "background-color: #f8d7da; border-left: 4px solid #dc3545; color: #721c24;"
                else:
                    rec_style = "background-color: #fff3cd; border-left: 4px solid #ffc107; color: #856404;"
                
                st.markdown(f"""
                <div style="{rec_style} padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h4 style="color: inherit; margin-bottom: 1rem; font-weight: bold;">📋 {recommendation['action']}</h4>
                    <p style="color: inherit; margin: 0.5rem 0; font-size: 1.1rem;"><strong>Confidence:</strong> {recommendation['confidence']}%</p>
                    <p style="color: inherit; margin: 0.5rem 0; font-size: 1.1rem;"><strong>Timeframe:</strong> {recommendation['timeframe']}</p>
                    <p style="color: inherit; margin: 0.5rem 0; font-size: 1.1rem; line-height: 1.5;"><strong>Reasoning:</strong> {recommendation['reasoning']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"Technical analysis failed: {tech_analysis['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def display_timeframe_analysis(timeframe_data, timeframe_name):
    """Display analysis for a specific timeframe"""
    if 'error' in timeframe_data:
        st.error(f"Error in {timeframe_name} analysis: {timeframe_data['error']}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend = timeframe_data['trend']
        trend_color = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "🟡"
        st.metric("Trend", f"{trend_color} {trend}")
    
    with col2:
        strength = timeframe_data['signal_strength']
        strength_color = "🟢" if "Strong" in strength else "🔴" if "Weak" in strength else "🟡"
        st.metric("Signal Strength", f"{strength_color} {strength}")
    
    with col3:
        resistance = timeframe_data['levels']['resistance']
        support = timeframe_data['levels']['support']
        st.metric("R/S Range", f"${support} - ${resistance}")
    
    # Technical indicators
    st.write("**📊 Technical Indicators:**")
    indicators = timeframe_data['indicators']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi = indicators['rsi']
        rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
        st.metric("RSI", f"{rsi_color} {rsi}")
    
    with col2:
        macd = indicators['macd']
        macd_color = "🟢" if macd > 0 else "🔴"
        st.metric("MACD", f"{macd_color} {macd}")
    
    with col3:
        stoch = indicators['stochastic_k']
        stoch_color = "🔴" if stoch > 80 else "🟢" if stoch < 20 else "🟡"
        st.metric("Stochastic", f"{stoch_color} {stoch}")
    
    with col4:
        if indicators['sma_20']:
            sma_20 = indicators['sma_20']
            st.metric("SMA 20", f"${sma_20}")

def show_live_trading_chart(symbol, tech_analysis, time_period="1M", chart_interval="1d", chart_type="Candlestick", show_volume=True, show_ma=True, show_signals=True):
    """Display live interactive chart with Grok-4 generated trading signals"""
    try:
        from data.fmp_market_data import fmp_provider
        import pandas as pd
        from datetime import datetime, timedelta
        import numpy as np
        
        # Convert time period to data points needed
        period_mapping = {
            "1D": 1, "5D": 5, "1M": 30, "3M": 90, 
            "6M": 180, "YTD": 250, "1Y": 365, "2Y": 730, "5Y": 1825
        }
        
        days_needed = period_mapping.get(time_period, 30)
        
        # Get historical data for chart using FMP as primary source
        try:
            # Use FMP API directly - more reliable
            data = fmp_provider.get_historical_data(symbol, f"{days_needed}d")
            
            if data.empty:
                # If FMP fails, create mock data for demonstration
                st.warning(f"⚠️ Unable to fetch real chart data for {time_period}. Using demo data for visualization.")
                
                # Create date range based on time period and interval
                if chart_interval in ["1m", "2m", "5m", "15m", "30m"]:
                    # For minute intervals, use hours as base
                    freq_map = {"1m": "1T", "2m": "2T", "5m": "5T", "15m": "15T", "30m": "30T"}
                    periods = min(days_needed * 24 * 60 // int(chart_interval.replace('m', '')), 1000)  # Limit to 1000 points
                    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq_map.get(chart_interval, "15T"))
                elif chart_interval in ["1h", "4h"]:
                    # For hourly intervals
                    freq_map = {"1h": "1H", "4h": "4H"}
                    periods = min(days_needed * 24 // int(chart_interval.replace('h', '')), 500)
                    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq_map.get(chart_interval, "1H"))
                elif chart_interval == "1wk":
                    periods = min(days_needed // 7, 100)
                    dates = pd.date_range(end=datetime.now(), periods=periods, freq="W")
                elif chart_interval == "1mo":
                    periods = min(days_needed // 30, 60)
                    dates = pd.date_range(end=datetime.now(), periods=periods, freq="M")
                else:  # 1d default
                    dates = pd.date_range(end=datetime.now(), periods=days_needed, freq='D')
                
                base_price = tech_analysis['current_price']
                
                # Create realistic mock data with appropriate volatility for timeframe
                np.random.seed(42)  # For consistent demo data
                
                # Adjust volatility based on timeframe
                if chart_interval in ["1m", "2m", "5m"]:
                    volatility = 0.005  # 0.5% for minute charts
                elif chart_interval in ["15m", "30m", "1h"]:
                    volatility = 0.01   # 1% for short-term charts
                elif chart_interval == "4h":
                    volatility = 0.015  # 1.5% for 4-hour charts
                else:  # daily and longer
                    volatility = 0.02   # 2% for daily charts
                
                price_changes = np.random.normal(0, base_price * volatility, len(dates))
                prices = [base_price]
                for change in price_changes[:-1]:
                    prices.append(max(prices[-1] + change, prices[-1] * 0.95))  # Prevent negative prices
                
                data = pd.DataFrame({
                    'Open': [p * (0.995 + np.random.random() * 0.01) for p in prices],
                    'High': [p * (1.005 + np.random.random() * 0.015) for p in prices],
                    'Low': [p * (0.985 + np.random.random() * 0.01) for p in prices],
                    'Close': prices,
                    'Volume': np.random.randint(500000, 15000000, len(dates))
                }, index=dates)
        except Exception as e:
            st.error(f"Error fetching chart data: {str(e)}")
            return
        
        if data.empty:
            st.error("Unable to fetch chart data")
            return
        
        # Create chart based on selected type
        fig = go.Figure()
        
        # Add main price chart based on chart type
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#1f77b4', width=2)
            ))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
        
        # Add volume bars if enabled
        if show_volume:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color='lightblue'
            ))
        
        # Add moving averages if enabled
        if show_ma:
            if len(data) >= 20:
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=2)
                ))
            
            if len(data) >= 50:
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='purple', width=2)
                ))
        
        # Add support and resistance levels
        current_price = tech_analysis['current_price']
        sr_levels = tech_analysis['support_resistance']
        
        # Resistance levels
        for resistance in sr_levels['resistance_levels'][:3]:  # Top 3 resistance levels
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color="red",
                annotation_text=f"R: ${resistance}",
                annotation_position="right"
            )
        
        # Support levels
        for support in sr_levels['support_levels'][:3]:  # Top 3 support levels
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color="green",
                annotation_text=f"S: ${support}",
                annotation_position="right"
            )
        
        # Add Grok-4 generated trading signals if enabled
        if show_signals:
            entry_signals = tech_analysis['entry_signals']
            best_signal = entry_signals.get('best_signal')
            
            if best_signal:
                # Current price marker
                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[current_price],
                    mode='markers',
                    name='Current Price',
                    marker=dict(
                        size=15,
                        color='blue',
                        symbol='circle',
                        line=dict(width=3, color='white')
                    )
                ))
                
                # Entry signal
                signal_color = 'green' if 'Long' in best_signal['type'] else 'red'
                signal_symbol = 'triangle-up' if 'Long' in best_signal['type'] else 'triangle-down'
                
                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[best_signal['entry_price']],
                    mode='markers+text',
                    name=f"Grok-4 {best_signal['type']} Signal",
                    marker=dict(
                        size=20,
                        color=signal_color,
                        symbol=signal_symbol,
                        line=dict(width=2, color='white')
                    ),
                    text=[f"{best_signal['type']} Entry"],
                    textposition="top center",
                    textfont=dict(size=12, color=signal_color)
                ))
                
                # Target price
                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[best_signal['target_price']],
                    mode='markers+text',
                    name='Target Price',
                    marker=dict(
                        size=15,
                        color='gold',
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Target: ${best_signal['target_price']}"],
                    textposition="top center",
                    textfont=dict(size=10, color='gold')
                ))
                
                # Stop loss
                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[best_signal['stop_loss']],
                    mode='markers+text',
                    name='Stop Loss',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Stop: ${best_signal['stop_loss']}"],
                    textposition="bottom center",
                    textfont=dict(size=10, color='red')
                ))
                
                # Add profit zone rectangle
                if 'Long' in best_signal['type']:
                    fig.add_shape(
                        type="rect",
                        x0=data.index[-5], x1=data.index[-1],
                        y0=best_signal['entry_price'], y1=best_signal['target_price'],
                        fillcolor="green",
                        opacity=0.1,
                        line_width=0,
                    )
                    # Add profit zone annotation
                    fig.add_annotation(
                        x=data.index[-3],
                        y=(best_signal['entry_price'] + best_signal['target_price']) / 2,
                        text=f"Profit Zone<br>${best_signal['potential_gain']:.2f} gain",
                        showarrow=False,
                        font=dict(size=10, color="green"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="green",
                        borderwidth=1
                    )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Live Chart with Grok-4 Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                range=[0, data['Volume'].max() * 4]
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified'
        )
        
        # Remove range slider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal summary below chart
        if best_signal:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entry Price", f"${best_signal['entry_price']}")
            with col2:
                st.metric("Target Price", f"${best_signal['target_price']}")
            with col3:
                st.metric("Stop Loss", f"${best_signal['stop_loss']}")
            with col4:
                st.metric("Risk/Reward", f"{best_signal['risk_reward_ratio']}:1")
            
            # Signal details
            st.info(f"🎯 **{best_signal['type']} Signal:** {best_signal['reasoning']}")
            st.success(f"💰 **Potential Gain:** ${best_signal['potential_gain']} | **Confidence:** {best_signal['confidence']}% | **Timeframe:** {best_signal['timeframe']}")
        
    except Exception as e:
        st.error(f"Error creating live chart: {str(e)}")
        st.info("📊 Chart data temporarily unavailable. Technical analysis continues to work normally.")

def show_cost_estimation(symbol):
    """Display Live Search cost estimation"""
    st.header(f"💰 Live Search Cost Estimation: {symbol}")
    
    try:
        cost_estimate = ai_engine.get_live_search_cost_estimate(symbol)
        
        # Cost overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Estimated Sources", cost_estimate['total_estimated_sources'])
        with col2:
            st.metric("Estimated Cost", f"${cost_estimate['estimated_cost_usd']:.2f}")
        with col3:
            st.metric("Cost per Source", f"${cost_estimate['cost_per_source']:.3f}")
        
        # Breakdown by analysis type
        st.subheader("📊 Cost Breakdown by Analysis Type")
        
        breakdown = cost_estimate['estimated_sources']
        breakdown_df = pd.DataFrame(list(breakdown.items()), columns=['Analysis Type', 'Sources'])
        breakdown_df['Cost'] = breakdown_df['Sources'] * cost_estimate['cost_per_source']
        
        fig = px.pie(breakdown_df, values='Sources', names='Analysis Type', 
                     title="Sources Distribution by Analysis Type")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost comparison table
        st.subheader("💵 Detailed Cost Analysis")
        breakdown_df['Cost (USD)'] = breakdown_df['Cost'].apply(lambda x: f"${x:.2f}")
        st.dataframe(breakdown_df[['Analysis Type', 'Sources', 'Cost (USD)']], use_container_width=True)
        
        # Cost optimization tips
        st.subheader("💡 Cost Optimization Tips")
        st.info("🔹 Use 'auto' mode to let Grok decide when Live Search is needed")
        st.info("🔹 Set max_search_results to limit sources per query")
        st.info("🔹 Focus on high-quality sources for better ROI")
        st.info("🔹 Consider time sensitivity - immediate decisions may cost more")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
