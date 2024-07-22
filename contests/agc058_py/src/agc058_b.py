import sys
import math

N = int(input())
P = list(map(int, input().split()))
MOD = 998244353
# [0, i)を見て、[左からjを渡す際のパターン数(=P[i-1]の値), 左にj以上を渡した際に得られるパターン数]
dp = [[[0]*(N+1), [0]*(N+1)] for _ in range(N+1)]
dp[0][0][0] = 1
# dp[0][0][0] = 0
for i in range(N):
    p = P[i]
    cur = 0
    # やりとりなし
    dp[i+1][0][p] = sum(dp[i][0][x] for x in range(N+1))%MOD
    # 左からもらう
    for j in range(p+1, N+1):
        dp[i+1][0][j] = dp[i][0][j]
    # 左に渡す
    dp[i+1][0][p] = (dp[i+1][0][p]+sum(dp[i][1][x] for x in range(p)))%MOD
    # ここでおしまい
    dp[i+1][1][p] = dp[i+1][0][p]
    # さらに渡す
    dp[i+1][1][p] = (dp[i+1][1][p]+sum(dp[i][1][x] for x in range(p+1)))%MOD
    # for y in range(i+1):
    #     dp[i+1][1][p] = (dp[i+1][1][p]+sum(dp[y][1][x]*max(i-y, 1) for x in range(p+1)))%MOD
    for j in range(p+1, N+1):
        dp[i+1][1][j] = dp[i][1][j]+dp[i][1][p]
for x in dp:
    print(x)
print(sum(dp[-1][0]))
