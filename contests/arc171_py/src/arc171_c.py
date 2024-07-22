import sys
import math
from collections import defaultdict

sys.setrecursionlimit(int(1e6))
N = int(input())
AB = [list(map(int, input().split())) for _ in range(N-1)]
MOD = 998244353
g = [[] for _ in range(N)]
for a, b in AB:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)
children = [[] for _ in range(N)]
parents = [None]*N
q = [(0, None)]
order = []
while len(q) > 0:
    u, p = q.pop()
    parents[u] = p
    order.append(u)
    for v in g[u]:
        if v != p:
            q.append((v, u))
            children[u].append(v)

kaizyo = [1]
for i in range(1, N+1):
    kaizyo.append((kaizyo[-1]*i)%MOD)
dp = [[0]*2 for _ in range(N)]
for u in order[::-1]:
    child_deg_cnt = [0]*(len(children[u])+1)
    child_deg_cnt[0] = 1
    for i, v in enumerate(children[u]):
        for j in range(i+1, -1, -1):
            if j>0:
                child_deg_cnt[j] = (child_deg_cnt[j]*dp[v][0]+child_deg_cnt[j-1]*dp[v][1])%MOD
            else:
                child_deg_cnt[j] = (child_deg_cnt[j]*dp[v][0])%MOD
    for deg, cnt in enumerate(child_deg_cnt):
        dp[u][0] = (dp[u][0]+cnt*kaizyo[deg])%MOD
        dp[u][1] = (dp[u][1]+cnt*kaizyo[deg+1])%MOD
print(dp[0][0])
