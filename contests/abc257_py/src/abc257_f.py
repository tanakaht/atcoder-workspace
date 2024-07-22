import sys
import math
import heapq

N, M = map(int, input().split())
UV = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
rest_g = []
for a, b in UV:
    a -= 1
    b -= 1
    if a==-1:
        rest_g.append(b)
    else:
        g[a].append((b, 1))
        g[b].append((a, 1))

start_node = 0
q = [(0, start_node)]
dists0 = [math.inf]*N
dists0[start_node] = 0
appeared = [False]*N
while q:
    d, u = heapq.heappop(q)
    if appeared[u]:
        continue
    appeared[u] = True
    for v, c in g[u]:
        d_ = d+c
        if dists0[v] > d_:
            dists0[v] = d_
            heapq.heappush(q, (d_, v))
start_node = N-1
q = [(0, start_node)]
distsN = [math.inf]*N
distsN[start_node] = 0
appeared = [False]*N
while q:
    d, u = heapq.heappop(q)
    if appeared[u]:
        continue
    appeared[u] = True
    for v, c in g[u]:
        d_ = d+c
        if distsN[v] > d_:
            distsN[v] = d_
            heapq.heappush(q, (d_, v))
if rest_g:
    minfrom0, minfromN = min([dists0[i] for i in rest_g]), min([distsN[i] for i in rest_g])
else:
    minfrom0, minfromN = math.inf, math.inf

anss = []
for i in range(N):
    ans = min([
        dists0[i]+1+minfromN,
        minfrom0+2+minfromN,
        minfrom0+1+distsN[i],
        dists0[N-1]
    ])
    if ans==math.inf:
        ans = -1
    anss.append(ans)
print(*anss)
