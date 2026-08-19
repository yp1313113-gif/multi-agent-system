# cache.py
import redis
from loguru import logger
from config import config

class Cache:
    def __init__(self):
        self.enabled = False
        self.client = None
        if config.CACHE_ENABLED:
            try:
                self.client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis 连接成功")
            except Exception as e:
                logger.warning(f"⚠️ Redis 连接失败: {e}")
                self.enabled = False
    
    def get(self, key):
        if not self.enabled:
            return None
        try:
            return self.client.get(f"rag_cache:{key}")
        except:
            return None
    
    def set(self, key, value):
        if not self.enabled:
            return False
        try:
            self.client.setex(f"rag_cache:{key}", 3600, value)
            return True
        except:
            return False

cache = Cache()