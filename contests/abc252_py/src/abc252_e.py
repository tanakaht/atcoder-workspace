import sys
import math
import heapq


N, M = map(int, input().split())
ABC = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
nodes2idx = {}
for i, (a, b, c) in enumerate(ABC):
    a -= 1
    b -= 1
    g[a].append((b, c))
    g[b].append((a, c))
    nodes2idx[(a, b)] = i
    nodes2idx[(b, a)] = i

start_node = 0
end_node = None
q = [(0, start_node, None)]
dists = [math.inf]*N
dists[start_node] = 0
appeared = [False]*N
anss = []
while q:
    d, u, fr = heapq.heappop(q)
    if appeared[u]:
        continue
    if fr is not None:
        anss.append(nodes2idx[u, fr])
    appeared[u] = True
    if u==end_node:
        # TODO: 処理
        break
    for v, c in g[u]:
        d_ = d+c
        if dists[v] > d_:
            dists[v] = d_
            heapq.heappush(q, (d_, v, u))
print(*map(lambda x:x+1, anss))
