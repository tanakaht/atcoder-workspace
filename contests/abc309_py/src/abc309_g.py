import sys
import math

N, X = map(int, input().split())
mod = 998244353
ans = 0
nanamekousei = [0]*(N+1) # 斜めの帯から縦横i個重複ないように取った時の取り方
dp = [[[0]*(1<<(2*X-1)) for _ in range(N+1)] for _ in range(N+1)] # j-1個目まで見て、k個取得済み、j-X..j+Xについて取得状況がbitになっているもの
dp[0][0][0] = 1
for j in range(N):
    for k in range(N+1):
        for bit in range((1<<(2*X-1))):
            # 取らない場合
            dp[j+1][k][bit>>1] = (dp[j][k][bit]+dp[j+1][k][bit>>1])%mod
            # 取る場合
            for l in range(j-X+1, j+X):
                if k==N:
                    continue
                if l<0 or l>=N:
                    continue
                # 取得済み
                if (bit>>(l-(j-X+1)))&1:
                    continue
                bit_ = bit | (1<<(l-(j-X+1)))
                dp[j+1][k+1][bit_>>1] = (dp[j][k][bit]+dp[j+1][k+1][bit_>>1])%mod
for i in range(N+1):
    nanamekousei[i] = sum(dp[-1][i])

kaizyo = [1]
for i in range(1, N+2):
    kaizyo.append((kaizyo[-1]*i)%mod)

ans = 0
for i in range(N+1):
    tmpans = (nanamekousei[i]*kaizyo[N-i])%mod
    if i%2:
        ans = (ans+tmpans)%mod
    else:
        ans = (ans-tmpans+mod)%mod

print((mod-ans)%mod)
