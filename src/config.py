MODEL_CONFIG = {
    "model_path": "/home/huongtt/.cache/huggingface/hub/"
                  "models--meta-llama--Llama-3.2-3B-Instruct/"
                  "snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "dtype": "float16",      
    "device_map": "auto",    
    "temperature": 0.7,
    "max_new_tokens": 256,
    "top_p": 0.9,
    "do_sample": True
}

AGENT_CONFIG = {
    "default_token_budget": 8000,
    "context_capacity": 32000,
    "cost_per_1k_tokens": 0.0,
    "alpha": 1.0,
    "beta": 1.0,
    "epsilon": 0.05,
    "confidence_threshold": 0.6,
    "gen_rate_hint": 180,
    "toolset": ["search", "summarize"]
}








