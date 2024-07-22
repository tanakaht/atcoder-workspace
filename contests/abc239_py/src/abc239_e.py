import sys
import math
from collections import defaultdict

sys.setrecursionlimit(int(1e6))
N, Q = map(int, input().split())
X = list(map(int, input().split()))
AB = [list(map(int, input().split())) for _ in range(N-1)]
VK = [list(map(int, input().split())) for _ in range(Q)]

g = [[] for _ in range(N)]
for a, b in AB:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)

children = [[] for _ in range(N)]
parents = [None]*N
q = [(0, None)]
while len(q) > 0:
    u, p = q.pop()
    parents[u] = p
    for v in g[u]:
        if v != p:
            q.append((v, u))
            children[u].append(v)

parent = [None]*N

dp1 = [None]*N
def dfs1(u):
    if dp1[u] is not None:
        return dp1[u]
    ret = [X[u]]
    for v in children[u]:
        for x in dfs1(v):
            ret.append(x)
    ret = sorted(ret)[::-1][:20]
    dp1[u] = ret
    return dp1[u]

dfs_order = []
appeared, withdrawed = [False]*N, [False]*N
start_node = 0
q = [start_node]
while q:
    u = q.pop()
    if appeared[u]:
        continue
    appeared[u] = True
    dfs_order.append(u)
    for v in g[u][::-1]:
        if appeared[v]:
            continue
        q.append(v)
for u in dfs_order[::-1]:
    dfs1(u)

for v, k in VK:
    print(dfs1(v-1)[k-1])
