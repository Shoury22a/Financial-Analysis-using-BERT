"""
Configuration Loader for FINSIGHT AI
Centralized configuration management using YAML and environment variables
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration manager"""
    
    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file and environment variables
        
        Args:
            config_path: Path to config.yaml (defaults to config/config.yaml)
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._override_with_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _override_with_env(self):
        """Override configuration with environment variables if present"""
        # API Keys
        if os.getenv('FINNHUB_API_KEY'):
            if 'api' not in self._config:
                self._config['api'] = {}
            self._config['api']['finnhub_key'] = os.getenv('FINNHUB_API_KEY')
        
        # Model settings
        if os.getenv('MODEL_NAME'):
            self._config['model']['name'] = os.getenv('MODEL_NAME')
        if os.getenv('MODEL_DIR'):
            self._config['model']['dir'] = os.getenv('MODEL_DIR')
        
        # Training settings
        if os.getenv('BATCH_SIZE'):
            self._config['training']['batch_size'] = int(os.getenv('BATCH_SIZE'))
        if os.getenv('LEARNING_RATE'):
            self._config['training']['learning_rate'] = float(os.getenv('LEARNING_RATE'))
        if os.getenv('NUM_EPOCHS'):
            self._config['training']['num_epochs'] = int(os.getenv('NUM_EPOCHS'))
        
        # API settings
        if os.getenv('API_HOST'):
            self._config['api']['host'] = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self._config['api']['port'] = int(os.getenv('API_PORT'))
        
        # Feature flags
        debug_logging = os.getenv('ENABLE_DEBUG_LOGGING', '').lower()
        if debug_logging in ['true', '1', 'yes']:
            self._config['logging']['level'] = 'DEBUG'
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('model.name')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self._config.get('training', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self._config.get('data', {})
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._config.get('api', {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self._config.get('ui', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config.get('logging', {})
    
    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_config = None

def get_config(config_path: str = None) -> Config:
    """
    Get or create global configuration instance
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = None):
    """
    Reload configuration from file
    
    Args:
        config_path: Optional path to config file
    """
    global _config
    _config = Config(config_path)
    return _config
