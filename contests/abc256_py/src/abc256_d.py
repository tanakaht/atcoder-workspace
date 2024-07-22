import sys
import math

N = int(input())
LRs = [list(map(int, input().split())) for _ in range(N)]
events = []
for l, r in LRs:
    events.append((l, 0))
    events.append((r+0.5, 1))
events = sorted(events)
ans = []
curl = None
curcnt = 0
for x, flg in events:
    if flg == 0:
        if curl is None:
            curl = x
        curcnt += 1
    if flg == 1:
        curcnt -= 1
        if curcnt == 0:
            ans.append((curl, int(x-0.5)))
            curl = None
for l, r in ans:
    print(l, r)
