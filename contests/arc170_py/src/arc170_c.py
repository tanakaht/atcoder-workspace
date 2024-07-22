import sys
import math

N, M = map(int, input().split())
S = list(map(int, input().split()))
MOD = 998244353
dp = [[0]*min(M+2, (N+1)) for _ in range(N+1)]
dp[0][0] = 1
for i in range(N):
    for j in range(min(M+2, i+1)):
        if S[i] == 1:
            if j<M+1:
                dp[i+1][j+1] = dp[i][j]
        else:
            dp[i+1][j] = (dp[i+1][j] + dp[i][j]*j)%MOD
            if j<M+1:
                dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]*(M-j))%MOD
print(sum(dp[-1])%MOD)
