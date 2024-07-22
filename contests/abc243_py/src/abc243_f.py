import sys
import math

N, M, K = map(int, input().split())
W = [int(input()) for _ in range(N)]
MOD = 998244353
dp = [[[0]*(M+1) for _ in range(K+1)] for _ in range(N+1)] #([0, i)の商品まで見て、 j回のくじを引いた, k種類の商品を取り出し済み)
dp[0][0][0] = 1
kaizyo = [1]
kaizyo_inv = [1]
tmp = 1
for i in range(1, 100):
    tmp = (tmp*i) % MOD
    kaizyo.append(tmp)
    kaizyo_inv.append(pow(tmp, MOD - 2, MOD))

def comb(n, r):
    if n < r or n < 0:
        return 0
    elif n == r or r==0:
        return 1
    else:
        return (((kaizyo[n] * kaizyo_inv[r])%MOD) * kaizyo_inv[n - r])%MOD

for i in range(1, N+1):
    for j in range(K+1):
        for k in range(M+1):
            dp[i][j][k] = dp[i-1][j][k]
            if k>0:
                for x in range(1, j+1):
                    dp[i][j][k] = (dp[i][j][k] + dp[i-1][j-x][k-1]*comb(j, x)*pow(W[i-1], x, MOD)) % MOD
ans = (dp[-1][-1][-1] * pow(pow(sum(W), K, MOD), MOD-2, MOD))%MOD
# print(dp[-1][-1][-1])
print(ans)
