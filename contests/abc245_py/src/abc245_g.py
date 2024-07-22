import sys
import math
import heapq

N, M, K, L = map(int, input().split())
A = list(map(lambda x: int(x)-1, input().split()))
B = list(map(lambda x: int(x)-1, input().split()))
UVC = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
for a, b, c in UVC:
    a -= 1
    b -= 1
    g[a].append((b, c))
    g[b].append((a, c))

def dijkstra(ninki):
    dist = [math.inf]*N
    appeared = [False]*N
    q = []
    for u in ninki:
        q.append((0, u))
        dist[u] = 0
    while q:
        d, u = heapq.heappop(q)
        if appeared[u]:
            continue
        appeared[u] = True
        for v, c in g[u]:
            if appeared[v]:
                continue
            d_ = d+c
            if d_ < dist[v]:
                dist[v] = d_
                heapq.heappush(q,d_, v))
    return dist


dists = {}  # (bitlength, 01) -> (nodeid -> cost)
for i in range(len(bin(K)[2:])+2):
    for flg in [0, 1]:
        # 該当する人気者を集める
        ninki = []
        for b in B:
            if ((A[b]>>i)&1)^flg:
                ninki.append(b)
        # dijkstra
        dists[(i, flg)] = dijkstra(ninki)
sys.exit(0)
anss = []
for u in range(N):
    ans = math.inf
    for i in range(len(bin(K)[2:])+2):
        flg = ((A[u]>>i)&1)
        ans = min(ans, dists[(i, flg)][u])
    anss.append(ans if ans != math.inf else -1)
print(*anss)
