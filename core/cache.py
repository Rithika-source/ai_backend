import diskcache as dc
from loguru import logger

# creates a cache folder automatically on disk
cache = dc.Cache("cache_store")

def get_cached(key: str):
    if key in cache:
        logger.info(f"Cache HIT for: '{key}'")
        return cache[key]
    logger.info(f"Cache MISS for: '{key}'")
    return None

def set_cache(key: str, value: str):
    cache.set(key, value, expire=3600)  # expires after 1 hour
    logger.info(f"Cache SET for: '{key}'")

def clear_cache():
    cache.clear()
    logger.info("Cache cleared")