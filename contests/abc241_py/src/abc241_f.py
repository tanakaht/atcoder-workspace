import sys
import math
from collections import defaultdict, deque
import bisect

H, W, N = map(int, input().split())
S = tuple(map(int, input().split()))
G = tuple(map(int, input().split()))
XY = [list(map(int, input().split())) for _ in range(N)]
xs, ys = defaultdict(list), defaultdict(list)
for x, y in XY:
    xs[x].append(y)
    ys[y].append(x)

for k, v in xs.items():
    xs[k] = sorted(v)
for k, v in ys.items():
    ys[k] = sorted(v)

def canmove(u):
    x, y = u
    ret = []
    yidx = bisect.bisect(xs[x], y)
    if yidx!=0:
        ret.append((x, xs[x][yidx-1]+1))
    if yidx!=len(xs[x]):
        ret.append((x, xs[x][yidx]-1))
    xidx = bisect.bisect(ys[y], x)
    if xidx!=0:
        ret.append((ys[y][xidx-1]+1, y))
    if xidx!=len(ys[y]):
        ret.append((ys[y][xidx]-1, y))
    return ret

q = deque([S])
appeared = defaultdict(lambda: False)
dist = defaultdict(lambda: math.inf)
dist[S] = 0
appeared[S] = True
while q:
    u = q.popleft()
    if u==G:
        print(dist[u])
        sys.exit(0)
    for v in canmove(u):
        if not appeared[v]:
            q.append(v)
            appeared[v] = True
            dist[v] = dist[u]+1
print(-1)
