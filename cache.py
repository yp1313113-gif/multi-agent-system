# cache.py
"""两级缓存：Redis（跨进程共享） + 进程内 LRU（兜底）。

━━━ 为什么要有第二级 ━━━
原实现只在 Redis 可用时才有缓存。但 Redis 不是永远在的（未部署、网络抖动、
容器重启中）——一旦连不上，缓存能力就**整体失效**，每个请求都要重新检索一遍，
延迟和成本直接翻倍。

这是典型的「单点依赖」问题。修法是加一层**进程内 LRU 兜底**：
  · Redis 可用  → 读写 Redis（多实例共享，容量大）
  · Redis 不可用 → 自动降级到进程内 LRU（单实例有效，但缓存能力不归零）

降级行为对上层完全透明（同样的 get/set 接口），属于**优雅降级**：
不是"要么全有要么全无"，而是"功能降级但服务不降级"。
"""
import time
from collections import OrderedDict

from loguru import logger

from config import config


class _LocalLRU:
    """进程内 LRU + TTL 兜底缓存（线程安全够用版：GIL 下 dict 操作原子）。"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        value, expire_at = item
        if expire_at and time.time() > expire_at:
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)   # LRU：命中即刷新热度
        self.hits += 1
        return value

    def set(self, key: str, value: str) -> bool:
        expire_at = time.time() + self.ttl if self.ttl else None
        self._data[key] = (value, expire_at)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:   # 超容量淘汰最久未用
            self._data.popitem(last=False)
        return True

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


class Cache:
    """两级缓存门面：优先 Redis，失败自动降级到进程内 LRU。"""

    def __init__(self):
        self.enabled = False
        self.client = None
        self.local = _LocalLRU(
            max_size=int(getattr(config, "CACHE_MAX_SIZE", 100)),
            ttl=config.REDIS_CACHE_TTL,
        )
        if config.CACHE_ENABLED and getattr(config, "REDIS_ENABLED", True):
            try:
                import redis
                self.client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    db=getattr(config, "REDIS_DB", 0),
                    password=getattr(config, "REDIS_PASSWORD", None),
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis 连接成功（一级缓存）")
            except Exception as e:
                logger.warning(f"⚠️ Redis 连接失败: {e} → 自动降级为进程内 LRU 缓存（二级缓存）")
                self.enabled = False
        else:
            logger.info("ℹ️ Redis 未启用 → 使用进程内 LRU 缓存（二级缓存）")

    @property
    def backend(self) -> str:
        """当前生效的缓存后端（可观测/面试演示用）。"""
        return "redis" if self.enabled else "local-lru"

    def get(self, key):
        if not config.CACHE_ENABLED:
            return None
        full_key = f"rag_cache:{key}"
        if self.enabled:
            try:
                val = self.client.get(full_key)
                if val is not None:
                    return val
                # Redis 未命中时再看本地（本地可能存了 Redis 故障期间的热点）
                return self.local.get(full_key)
            except Exception as e:
                logger.warning(f"⚠️ Redis 读取失败: {e} → 降级本地缓存")
                self.enabled = False
        return self.local.get(full_key)

    def set(self, key, value):
        if not config.CACHE_ENABLED:
            return False
        full_key = f"rag_cache:{key}"
        # 本地永远写（保证 Redis 抖动期间新结果也能命中）
        self.local.set(full_key, value)
        if self.enabled:
            try:
                self.client.setex(full_key, config.REDIS_CACHE_TTL, value)
                return True
            except Exception as e:
                logger.warning(f"⚠️ Redis 写入失败: {e} → 降级本地缓存")
                self.enabled = False
        return True

    def stats(self) -> dict:
        """缓存可观测：当前后端 + 命中率（写进 /health 便于演示）。"""
        return {"backend": self.backend, "enabled": config.CACHE_ENABLED, "local": self.local.stats()}


cache = Cache()
