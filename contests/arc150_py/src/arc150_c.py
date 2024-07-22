import sys
import math
import heapq

N, M, K = map(int, input().split())
UV = [list(map(int, input().split())) for _ in range(M)]
A = list(map(int, input().split()))
B = list(map(int, input().split()))
g = [[] for _ in range(N)]
for a, b in UV:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)

start_node = 0
end_node = N-1
q = [(int(B[0]==A[start_node]), start_node)]
dists = [math.inf]*N
dists[start_node] = 0
appeared = [False]*N
while q:
    d, u = heapq.heappop(q)
    if d==K:
        break
    if appeared[u]:
        continue
    appeared[u] = True
    if u==end_node:
        break
    for v in g[u]:
        d_ = d+(B[d]==A[v])
        if dists[v] > d_:
            dists[v] = d_
            heapq.heappush(q, (d_, v))

if dists[N-1]<K:
    print("No")
else:
    print("Yes")
