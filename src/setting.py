# Cấu hình của vectorstore
VECTORSTORE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "chunk_type": "character",
    "k": 7,
    "separators": ["\n\n", "\n", ". ", "! ", "? ", ":", ";", " "],
    "embedding_type": "huggingface",
    "embedding_model": "BAAI/bge-base-en",
    "enable_hybrid_search": False,
    "separator": "\n\n",
}

# Cấu hình của WDMParser
IGNORE_TABLES = True
ENRICH_TABLES = False

# Cấu hình của Reranker
USE_REANKER = True
REANKER_MODEL_NAME = "bce"


# Cấu hình của LLM
QUERY_ANALYSIS_MODEL = {
    "model_provider": "google_vertexai",
    "model_name": "gemini-2.0-flash",
}

QUERY_ANALYSIS_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 1024,
}

# ============================== CONVERSATION OPTIMIZATION CONFIG ==============================

# Default conversation optimization configuration
CONVERSATION_OPTIMIZATION_CONFIG = {
    # Token Management
    "max_conversation_tokens": 4000,        # Total tokens allowed in conversation context
    "conversation_token_buffer": 500,       # Buffer for response generation
    "max_recent_messages": 10,              # Maximum recent messages to keep
    "summary_ratio": 0.3,                   # Ratio of tokens allocated to summary vs recent messages
    
    # Summarization Settings
    "summarize_after": 20,                  # Trigger summarization after N messages
    "summarization_model": "gemini-2.0-flash", # LLM model for summarization
    "summarization_temperature": 0.1,       # Temperature for consistent summaries
    "max_summary_tokens": 512,              # Maximum tokens for generated summaries
    
    # Performance Settings
    "enable_optimization": True,             # Enable/disable optimization globally
    "enable_async_summarization": True,     # Use async summarization when possible
    "enable_semantic_compression": False,   # Future: semantic similarity compression
    "enable_priority_retention": False,     # Future: importance-based message retention
    
    # Fallback Settings
    "fallback_to_character_count": True,    # Use character count if tiktoken fails
    "character_to_token_ratio": 4,          # Approximation: 4 chars ≈ 1 token
    "min_messages_to_optimize": 5,          # Minimum messages before optimization kicks in
    
    # Debug and Monitoring
    "enable_metrics_tracking": True,        # Track optimization metrics
    "log_optimization_decisions": True,     # Log when optimizations are applied
    "debug_token_counting": False,          # Detailed logging for token counting
}

# Different presets for various use cases
CONVERSATION_OPTIMIZATION_PRESETS = {
    "conservative": {
        **CONVERSATION_OPTIMIZATION_CONFIG,
        "max_conversation_tokens": 2000,
        "conversation_token_buffer": 300,
        "max_recent_messages": 6,
        "summarize_after": 12,
        "summary_ratio": 0.2,
    },
    
    "default": CONVERSATION_OPTIMIZATION_CONFIG,
    
    "aggressive": {
        **CONVERSATION_OPTIMIZATION_CONFIG,
        "max_conversation_tokens": 6000,
        "conversation_token_buffer": 800,
        "max_recent_messages": 15,
        "summarize_after": 30,
        "summary_ratio": 0.4,
    },
    
    "mobile_optimized": {
        **CONVERSATION_OPTIMIZATION_CONFIG,
        "max_conversation_tokens": 1500,
        "conversation_token_buffer": 200,
        "max_recent_messages": 5,
        "summarize_after": 8,
        "summary_ratio": 0.25,
    },
    
    "development": {
        **CONVERSATION_OPTIMIZATION_CONFIG,
        "enable_optimization": True,
        "debug_token_counting": True,
        "log_optimization_decisions": True,
        "summarize_after": 6,  # Quick testing
        "max_recent_messages": 4,
    },
    
    "production": {
        **CONVERSATION_OPTIMIZATION_CONFIG,
        "enable_optimization": True,
        "debug_token_counting": False,
        "log_optimization_decisions": False,
        "enable_metrics_tracking": True,
    }
}

# Model-specific token limits
MODEL_TOKEN_LIMITS = {
    "gemini-2.0-flash": {
        "max_context": 32768,
        "recommended_conversation_tokens": 4000,
        "buffer": 500,
    },
    "gemini-1.5-pro": {
        "max_context": 2097152,  # 2M tokens
        "recommended_conversation_tokens": 8000,
        "buffer": 1000,
    },
    "gpt-4": {
        "max_context": 8192,
        "recommended_conversation_tokens": 3000,
        "buffer": 500,
    },
    "gpt-4-turbo": {
        "max_context": 128000,
        "recommended_conversation_tokens": 6000,
        "buffer": 1000,
    },
    "claude-3": {
        "max_context": 200000,
        "recommended_conversation_tokens": 8000,
        "buffer": 1000,
    }
}

def get_conversation_optimization_config(preset_name: str = "default") -> dict:
    """
    Get configuration for conversation optimization
    
    Args:
        preset_name: Name of the preset to use
        
    Returns:
        dict: Configuration dictionary
    """
    if preset_name not in CONVERSATION_OPTIMIZATION_PRESETS:
        print(f"Warning: Unknown preset '{preset_name}', using 'default'")
        preset_name = "default"
    
    return CONVERSATION_OPTIMIZATION_PRESETS[preset_name].copy()

def get_model_conversation_config(model_name: str) -> dict:
    """Get model-specific conversation configuration"""
    if model_name in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model_name]
    
    # Default fallback
    return MODEL_TOKEN_LIMITS["gemini-2.0-flash"]

def create_optimized_conversation_config(model_name: str = "gemini-2.0-flash", 
                                        use_case: str = "default",
                                        custom_overrides: dict | None = None) -> dict:
    """
    Create an optimized conversation configuration
    
    Args:
        model_name: Name of the LLM model to optimize for
        use_case: Use case preset (conservative, default, aggressive, etc.)
        custom_overrides: Custom settings to override defaults
        
    Returns:
        dict: Optimized configuration
    """
    # Start with use case preset
    config = get_conversation_optimization_config(use_case)
    
    # Apply model-specific optimizations
    model_config = get_model_conversation_config(model_name)
    config.update({
        "max_conversation_tokens": model_config["recommended_conversation_tokens"],
        "conversation_token_buffer": model_config["buffer"],
        "summarization_model": model_name,
    })
    
    # Apply custom overrides if provided
    if custom_overrides is not None:
        config.update(custom_overrides)
    
    return config