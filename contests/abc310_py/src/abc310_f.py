import sys
import math

N = int(input())
A = list(map(int, input().split()))
mod = 998244353
A_inv = [pow(a, mod-2, mod) for a in A]
dp = [[0]*(1<<11) for _ in range(N+1)]
dp[0][1] = 1

for i in  range(N):
    for j in range(1<<11):
        for x in range(1, min(10, A[i])+1):
            j_ = j
            for k in range(11):
                if (j>>k)&1 and k+x<=10:
                    j_ |= (1<<(k+x))
            dp[i+1][j_] = (dp[i+1][j_]+dp[i][j]*A_inv[i])%mod
        dp[i+1][j] = (dp[i+1][j]+dp[i][j]*max(0, A[i]-10)*A_inv[i])%mod

ans = 0
for j in range(1<<11):
    if (j>>10)&1:
        ans = (ans+dp[-1][j])%mod
print(ans)
