# -*- coding: utf-8 -*-
import io, sys, asyncio, time, aiohttp
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = 'http://127.0.0.1:8001/chat'
MSG = '加计扣除比例是多少？'

async def one(session, idx, results):
    t0 = time.perf_counter()
    first = None
    nbytes = 0
    try:
        async with session.get(BASE, params={'message': MSG, 'session': f'test{idx}'}) as r:
            st = r.status
            if st == 200:
                async for chunk in r.content.iter_any():
                    if first is None:
                        first = time.perf_counter() - t0
                    nbytes += len(chunk)
            else:
                await r.read()
            results.append(dict(idx=idx, status=st, ttfb=first, total=time.perf_counter()-t0, bytes=nbytes))
    except Exception as e:
        results.append(dict(idx=idx, status=-1, ttfb=None, total=time.perf_counter()-t0, bytes=0, err=str(e)[:40]))

async def main():
    N = 15
    results = []
    conn = aiohttp.TCPConnector(limit=30)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        t0 = time.perf_counter()
        await asyncio.gather(*[one(s, i, results) for i in range(N)])
        wall = time.perf_counter() - t0
    ok = [r for r in results if r['status']==200]
    busy = [r for r in results if r['status']==503]
    other = [r for r in results if r['status'] not in (200,503)]
    print('=' * 60)
    print(f'端到端压测：/chat 接口，{N} 个并发同时提问')
    print('=' * 60)
    print(f'  总墙钟耗时 : {wall:.1f} 秒')
    print(f'  成功 (200) : {len(ok)}')
    print(f'  限流 (503) : {len(busy)}')
    print(f'  其他       : {len(other)}')
    if ok:
        ttfb = [r['ttfb'] for r in ok if r['ttfb']]
        tot  = [r['total'] for r in ok]
        bts  = [r['bytes'] for r in ok]
        print(f'  首字节 P50 : {sorted(ttfb)[len(ttfb)//2]:.1f} 秒')
        print(f'  首字节 最快 : {min(ttfb):.1f} 秒' if ttfb else '')
        print(f'  完成   P50 : {sorted(tot)[len(tot)//2]:.1f} 秒')
        print(f'  返回字节 P50: {sorted(bts)[len(bts)//2]} 字节')
    print('=' * 60)
    print()
    print('逐个结果：')
    for r in sorted(results, key=lambda x: x['idx']):
        st = '200 成功' if r['status']==200 else ('503 限流' if r['status']==503 else str(r['status']))
        ttfb = f"{r['ttfb']:.1f}s" if r.get('ttfb') else '  -  '
        print(f"  #{r['idx']:<3} {st:<10} 首字节 {ttfb:<8} 总耗时 {r['total']:.1f}s")

asyncio.run(main())