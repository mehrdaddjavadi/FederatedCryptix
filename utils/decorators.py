import json
import functools

def load_model_config(config_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            return func(*args, model_config=model_config, **kwargs)
        return wrapper
    return decorator

def load_training_config(config_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            return func(*args, training_config=training_config, **kwargs)
        return wrapper
    return decorator
