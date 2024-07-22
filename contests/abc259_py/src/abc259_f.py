import sys
import math
from collections import defaultdict

sys.setrecursionlimit(int(1e6))

N = int(input())
D = list(map(int, input().split()))
UVW = [list(map(int, input().split())) for _ in range(N-1)]

g = [[] for _ in range(N)]
for a, b, w in UVW:
    a -= 1
    b -= 1
    g[a].append((b, max(0, w)))
    g[b].append((a, max(0, w)))

children = [[] for _ in range(N)]
parents = [None]*N
q = [(0, None, 0)]
dfs_ord = []
while len(q) > 0:
    u, p, w = q.pop()
    parents[u] = (p, w)
    dfs_ord.append(u)
    for v, w_ in g[u]:
        if v != p:
            q.append((v, u, w_))
            children[u].append((v, w_))

dp1 = [None]*N
# (親とのパスを取るときの最大, 親とのパスを取らない時の最大)
def dfs1(u):
    if dp1[u] is not None:
        return dp1[u]
    p = parents[u][0]
    childs_d = []
    for v, w in children[u]:
        if v == p:
            continue
        x, y = dfs1(v)
        childs_d.append((x-y, x, y))
    childs_d = sorted(childs_d)[::-1]
    ret = [parents[u][1], 0]
    for i, (_, x, y) in enumerate(childs_d):
        if i<D[u]-1:
            ret[0] += x
            ret[1] += x
        elif i==D[u]-1:
            ret[0] += y
            ret[1] += x
        else:
            ret[0] += y
            ret[1] += y
    # print(u, childs_d, children[u])
    if D[u] == 0:
        ret[0] = ret[1]
    else:
        ret[0] = max(ret[0], ret[1])
    dp1[u] = ret
    return dp1[u]

for u in dfs_ord[::-1]:
    dfs1(u)
print(dfs1(0)[0])
# print(dp1)
