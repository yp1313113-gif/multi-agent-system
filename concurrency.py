# concurrency.py
"""并发控制模块：异步限流（asyncio.Semaphore）+ 缓存防击穿（single-flight）。

背景：Agent 应用面向大量用户时，直接放开并发会把 LLM API 打限流（429）、
把数据库打爆。这里用两层保护，属于「高并发/分布式」工程素养的轻量落地：

  1) AsyncLimiter —— 并发限流：限制同时执行的 LLM 请求数（Semaphore）。
     并发满时排队等待，排队超时抛 BusyError → API 返回 503「系统繁忙」。
     保护下游（LLM / 数据库），而不是硬扛到崩。

  2) SyncSingleFlight —— 缓存防击穿：热点问题缓存过期瞬间，100 个并发
     请求同时 miss 会 100 次穿透到 LLM/数据库（缓存击穿）。
     同一 key 只让第一个请求去重建，其余请求轮询等待后直接读缓存结果，
     共享一次重建（single-flight 单飞）。

用法：
  from concurrency import limiter, single_flight, BusyError

  # API 层限流
  if not await limiter.acquire():
      return JSONResponse(status_code=503, ...)
  try:
      ...  # 重活（LLM 调用）
  finally:
      limiter.release()

  # 缓存防击穿
  result = single_flight.run(key, rebuild_fn, read_back_fn)
"""
import asyncio
import threading
import time
from loguru import logger
from config import config


class BusyError(Exception):
    """并发上限已满且排队超时（调用方应返回 503 类响应）。"""


class AsyncLimiter:
    """异步信号量限流：并发满时有限排队，超时抛 BusyError。"""

    def __init__(self, max_concurrency: int = 10, queue_timeout: float = 5.0):
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self._sem = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> bool:
        """尝试获取一个并发名额。排队超时返回 False（不再抛异常，调用方好处理）。"""
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"⏳ 并发已满（>={self.max_concurrency}），排队 {self.queue_timeout}s 超时，拒绝请求"
            )
            return False

    def release(self):
        self._sem.release()

    @property
    def running(self) -> int:
        """当前已占用名额数（调试/监控用）。"""
        return self.max_concurrency - self._sem._value


class SyncSingleFlight:
    """同步版 single-flight：同一 key 并发 miss 时只重建一次，其余等待共享结果。

    实现：key → 互斥锁。拿不到锁 = 别人正在重建 → 轮询 read_back()
    （重建者已把结果写入缓存，等待者直接读缓存，不重复调用 LLM/数据库）。
    """

    def __init__(self, rebuild_wait: float = 0.05, max_wait: float = 8.0):
        self._locks: dict = {}
        self._guard = threading.Lock()
        self.rebuild_wait = rebuild_wait
        self.max_wait = max_wait
        self._rebuilt_count = 0       # 统计实际重建次数（压测/面试用）
        self._waited_count = 0        # 统计被共享结果的等待次数

    def run(self, key: str, rebuild, read_back):
        """以 key 为粒度执行：
        - 无人重建该 key → 执行 rebuild() 并返回；
        - 有人正在重建 → 轮询 read_back() 拿重建后的结果（不重复重建）。
        """
        lock = self._get_lock(key)
        if lock.acquire(blocking=False):
            try:
                self._rebuilt_count += 1
                return rebuild()
            finally:
                lock.release()
                self._cleanup(key)

        # 别人正在重建：轮询等待结果
        deadline = time.time() + self.max_wait
        while time.time() < deadline:
            time.sleep(self.rebuild_wait)
            value = read_back()
            if value is not None:
                self._waited_count += 1
                return value
        # 等待超时（重建者失败）：降级为自行重建，保证可用性
        logger.warning(f"⚠️ [{key}] single-flight 等待超时，降级为自行重建")
        return rebuild()

    def _get_lock(self, key: str) -> threading.Lock:
        with self._guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _cleanup(self, key: str):
        with self._guard:
            self._locks.pop(key, None)

    def stats(self) -> dict:
        """重建/等待统计（可观测、可写进 README）。"""
        return {"rebuilt": self._rebuilt_count, "waited": self._waited_count}


# 全局单例（按 config 初始化）
limiter = AsyncLimiter(
    max_concurrency=config.MAX_CONCURRENCY,
    queue_timeout=config.CONCURRENCY_QUEUE_TIMEOUT,
)
single_flight = SyncSingleFlight()
