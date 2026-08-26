"""并发控制模块测试：AsyncLimiter 限流 + SyncSingleFlight 防击穿。

覆盖语义（面试可讲）：
  1) 并发上限强制：同时运行的任务数不超过 max_concurrency；
  2) 排队超时拒绝：并发满且排队超时 → acquire 返回 False（API 层转 503）；
  3) 同一 key 并发 miss 只重建一次（single-flight），其余共享结果；
  4) 不同 key 互不阻塞（key 粒度并发）；
  5) 等待超时降级为自行重建（保证可用性）。
"""
import os
import sys
import time
import asyncio
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrency import AsyncLimiter, SyncSingleFlight


class TestAsyncLimiter:
    def test_max_concurrency_enforced(self):
        """并发 5 个任务，同时运行数 ≤ 2，且全部成功。"""
        lim = AsyncLimiter(max_concurrency=2, queue_timeout=5)
        running = 0
        peak = 0
        lock = threading.Lock()

        async def worker():
            nonlocal running, peak
            if not await lim.acquire():
                return "busy"
            try:
                with lock:
                    running += 1
                    peak = max(peak, running)
                await asyncio.sleep(0.05)
                with lock:
                    running -= 1
                return "ok"
            finally:
                lim.release()

        async def main():
            results = await asyncio.gather(*[worker() for _ in range(5)])
            return results, peak

        results, peak = asyncio.run(main())
        assert peak <= 2, f"峰值并发 {peak} 超过上限 2"
        assert all(r == "ok" for r in results)

    def test_queue_timeout_rejects(self):
        """并发满且排队超时 → acquire 返回 False（调用方转 503）。"""
        lim = AsyncLimiter(max_concurrency=1, queue_timeout=0.1)

        async def main():
            ok1 = await lim.acquire()
            ok2 = await lim.acquire()   # 名额已满，0.1s 超时
            lim.release()
            return ok1, ok2

        ok1, ok2 = asyncio.run(main())
        assert ok1 is True
        assert ok2 is False


class TestSyncSingleFlight:
    def test_same_key_rebuilds_once(self):
        """10 个线程同时打同一 key：重建只发生 1 次，全部拿到结果。"""
        sf = SyncSingleFlight(rebuild_wait=0.01, max_wait=2)
        rebuild_count = 0
        store = {}
        lock = threading.Lock()

        def rebuild(key):
            nonlocal rebuild_count
            with lock:
                rebuild_count += 1
            time.sleep(0.1)          # 模拟慢重建（LLM 生成）
            store[key] = f"v:{key}"
            return store[key]

        def read_back(key):
            return store.get(key)

        def worker(key):
            return sf.run(key, lambda: rebuild(key), lambda: read_back(key))

        threads = [threading.Thread(target=worker, args=("k1",)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert rebuild_count == 1, f"重建了 {rebuild_count} 次，应只有 1 次"
        assert sf.stats()["waited"] >= 9  # 其余 9 个都在等待共享结果

    def test_different_keys_not_blocked(self):
        """不同 key 互不阻塞（key 粒度并发）。"""
        sf = SyncSingleFlight(rebuild_wait=0.01, max_wait=2)
        store = {}

        def slow_rebuild(key):
            time.sleep(0.2)
            store[key] = "slow"
            return "slow"

        def read_back(key):
            return store.get(key)

        r1 = sf.run("a", lambda: slow_rebuild("a"), lambda: read_back("a"))
        r2 = sf.run("b", lambda: (store.__setitem__("b", "fast"), "fast")[1], lambda: read_back("b"))
        assert r1 == "slow"
        assert r2 == "fast"

    def test_wait_timeout_falls_back_to_rebuild(self):
        """重建者卡住且不写缓存 → 等待者超时后自行重建（可用性兜底）。"""
        sf = SyncSingleFlight(rebuild_wait=0.005, max_wait=0.05)
        calls = []
        lock = threading.Lock()
        holder_entered = threading.Event()

        def rebuild():
            with lock:
                calls.append(1)
                n = len(calls)
            if n == 1:
                holder_entered.set()
                time.sleep(0.3)      # 第一个重建者卡住、不写缓存
            return f"r{n}"

        def read_back():
            return None              # 永远读不到 → 触发超时降级

        def worker():
            return sf.run("k", rebuild, read_back)

        t1 = threading.Thread(target=worker)
        t1.start()
        holder_entered.wait()        # 等第一个占用锁
        t2 = threading.Thread(target=worker)
        t2.start()
        t2.join(timeout=1)
        t1.join(timeout=1)

        assert len(calls) == 2       # 第二个等待超时后自行重建
