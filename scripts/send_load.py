"""Send paced HTTP requests through the live gateway during the Kubernetes demo."""
import argparse
import asyncio
import time
import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/hit")
    p.add_argument("--rps", type=float, default=120)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--workers", type=int, default=80)
    args = p.parse_args()
    total = int(args.rps * args.duration)
    async def paced_run():
        start = time.monotonic()
        sem = asyncio.Semaphore(args.workers)
        async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)) as client:
            async def send(i):
                delay = start + i / args.rps - time.monotonic()
                if delay > 0: await asyncio.sleep(delay)
                async with sem:
                    try:
                        response = await client.post(args.url)
                        return response.status_code == 200
                    except httpx.HTTPError:
                        return False
            results = await asyncio.gather(*(send(i) for i in range(total)))
        elapsed = time.monotonic() - start
        print(f"sent={total} errors={results.count(False)} elapsed_seconds={elapsed:.1f} achieved_rps={total/elapsed:.1f}")
    asyncio.run(paced_run())


if __name__ == "__main__": main()
