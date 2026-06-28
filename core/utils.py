import json
import hashlib
import numpy as np

def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize(v) for v in obj]
    elif isinstance(obj, float):
        return round(obj, 4)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return _normalize(obj.tolist())
    else:
        return obj

def hash_config(config: dict) -> str:
    normalized = _normalize(config)
    canonical = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]

import numpy as np

def clean_for_json(obj):
    """Recursively convert numpy types to Python built-ins for JSON serialization."""
    if isinstance(obj, dict):
        return {clean_for_json(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    else:
        return obj