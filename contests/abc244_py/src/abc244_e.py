import sys
import math

N, M, K, S, T, X = map(int, input().split())
UV = [list(map(int, input().split())) for _ in range(M)]
dp = [[[0]*N for _ in range(K+1)] for _ in range(2)] # xの出場回数, i回移動した, 今node j->パターン数
MOD = 998244353
g = [[] for _ in range(N)]
for a, b in UV:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)
X -= 1
dp[0][0][S-1] = 1
for i in range(K):
    for u in range(N):
        for f in range(2):
            for v in g[u]:
                dp[f][i+1][v] = (dp[f][i+1][v]+dp[f][i][u])%MOD
    dp[0][i+1][X], dp[1][i+1][X] = dp[1][i+1][X], dp[0][i+1][X]
print(dp[0][-1][T-1])
