import sys
import math

N, M = map(int, input().split())
UVW = [list(map(int, input().split())) for _ in range(M)]

g = [[] for _ in range(N)]
for u, v, w in UVW:
    u -= 1
    v -= 1
    g[u].append((v, w))

dist = [[math.inf] * N for _ in range(N)]
for i in range(N):
    dist[i][i] = 0
    for j, d in g[i]:
        dist[i][j] = d
for k in range(N):
    for i in range(N):
        for j in range(N):
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

dp = [[math.inf] * (1 << N) for _ in range(N)] # 今iにいて、訪れた頂点の集合がSであるときの最小コスト
for i in range(N):
    dp[i][1 << i] = 0
for S in range(1<<N):
    for i in range(N):
        cost = dp[i][S]
        for j in range(N):
            nS = S | (1<<j)
            if nS!=S:
                ncost = cost + dist[i][j]
                dp[j][nS] = min(dp[j][nS], ncost)
ans = min(dp[i][(1 << N) - 1] for i in range(N))
print(ans if ans != math.inf else "No")
