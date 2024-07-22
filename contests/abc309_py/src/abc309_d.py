import sys
import math
import heapq

N1, N2, M = map(int, input().split())
AB = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N1+N2)]
for a, b in AB:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)

start_node = 0
end_node = None
q = [(0, start_node)]
dists = [math.inf]*(N1+N2)
dists[start_node] = 0
appeared = [False]*(N1+N2)
while q:
    d, u = heapq.heappop(q)
    if appeared[u]:
        continue
    appeared[u] = True
    if u==end_node:
        # TODO: 処理
        break
    for v in g[u]:
        d_ = d+1
        if dists[v] > d_:
            dists[v] = d_
            heapq.heappush(q, (d_, v))

max_dist1 = max(dists[:N1])

start_node = N1+N2-1
end_node = None
q = [(0, start_node)]
dists = [math.inf]*(N1+N2)
dists[start_node] = 0
appeared = [False]*(N1+N2)
while q:
    d, u = heapq.heappop(q)
    if appeared[u]:
        continue
    appeared[u] = True
    if u==end_node:
        # TODO: 処理
        break
    for v in g[u]:
        d_ = d+1
        if dists[v] > d_:
            dists[v] = d_
            heapq.heappush(q, (d_, v))

max_dist2 = max(dists[N1:])
print(max_dist1+max_dist2+1)
