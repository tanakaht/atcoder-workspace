import sys
import math

N = int(input())
A = list(map(int, input().split()))
mod = 998244353
A_inv = [pow(a, mod-2, mod) for a in A]
dp = [[0]*(1<<10) for _ in range(N+1)]
dp[0][0] = 1

for i in  range(N):
    for j in range(1<<10):
        for x in range(min(10, A[i])):
            dp[i+1][j|(1<<x)] = (dp[i+1][j|(1<<x)]+dp[i][j]*A_inv[i])%mod
        dp[i+1][j] = (dp[i+1][j]+dp[i][j]*max(0, A[i]-10)*A_inv[i])%mod

ans = 0
for j in range(1<<10):

    available = set([0])
    for x in [x+1 for x in range(10) if (j>>x)&1]:
        for v in list(available):
            if x+v<=10:
                available.add(x+v)
    if 10 in available:
        if dp[-1][j]!=0:
            print(dp[-1][j]*126%mod, [x+1 for x in range(10) if (j>>x)&1])
        ans = (ans+dp[-1][j])%mod
print(ans)
