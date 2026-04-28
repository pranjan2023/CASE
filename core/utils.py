import json
import hashlib

def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize(v) for v in obj]
    elif isinstance(obj, float):
        return round(obj, 4)   # critical
    else:
        return obj

def hash_config(config: dict) -> str:
    normalized = _normalize(config)
    canonical = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]