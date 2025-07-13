"""
API Keys and Credentials Configuration
Store your API keys here or use environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

class APIKeys:
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # YouTube Data API
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    
    # Interactive Brokers
    IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
    IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))  # 7497 for paper trading, 7496 for live
    IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
    
    # AI APIs
    XAI_API_KEY = os.getenv('XAI_API_KEY', '')  # Grok-3 API key
    
    @classmethod
    def validate_keys(cls):
        """Validate that required API keys are present"""
        required_keys = [
            ('SUPABASE_URL', cls.SUPABASE_URL),
            ('SUPABASE_KEY', cls.SUPABASE_KEY),
            ('YOUTUBE_API_KEY', cls.YOUTUBE_API_KEY),
        ]
        
        missing_keys = []
        for key_name, key_value in required_keys:
            if not key_value:
                missing_keys.append(key_name)
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
