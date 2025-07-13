"""
Supabase Database Configuration and Schema
"""
from supabase import create_client, Client
from config.api_keys import APIKeys
import logging

class SupabaseManager:
    def __init__(self):
        self.client: Client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Supabase client"""
        try:
            APIKeys.validate_keys()
            self.client = create_client(
                supabase_url=APIKeys.SUPABASE_URL, 
                supabase_key=APIKeys.SUPABASE_KEY
            )
            logging.info("Supabase client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def create_tables(self):
        """Create all required tables for the trading system"""
        # Note: Tables should be created via Supabase dashboard or SQL editor
        # This method is for reference only
        logging.info("Tables should be created via Supabase dashboard. Skipping table creation.")
        pass
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);",
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe);",
            "CREATE INDEX IF NOT EXISTS idx_sentiment_data_symbol ON sentiment_data(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_state_symbol ON portfolio_state(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_ai_memory_embedding ON ai_memory USING ivfflat (embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_etf_holdings_etf_symbol ON etf_holdings(etf_symbol);",
            "CREATE INDEX IF NOT EXISTS idx_options_data_symbol_exp ON options_data(symbol, expiration_date);"
        ]
        
        for sql in indexes_sql:
            try:
                self.client.rpc('execute_sql', {'sql': sql}).execute()
                logging.info(f"Index created successfully: {sql[:50]}...")
            except Exception as e:
                logging.warning(f"Index creation failed (may already exist): {e}")

# Global instance
supabase_manager = SupabaseManager()
