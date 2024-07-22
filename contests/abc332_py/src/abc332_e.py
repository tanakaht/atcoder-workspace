import sys
import math

N, D = map(int, input().split())
W = list(map(int, input().split()))
dp = [[math.inf]*(1<<N) for _ in range(D+1)]
xmean = sum(W)/D
for x in range(1<<N):
    tmp = 0
    for i in range(N):
        if x & (1<<i):
            tmp += W[i]
    tmp -= xmean
    dp[1][x] = (tmp*tmp)/D
for d in range(D):
    for flg1 in range(1<<N):
        flg2 = flg1
        while flg2 > 0:
            flg2 = (flg2-1)&flg1
            dp[d+1][flg1] = min(dp[d+1][flg1], dp[d][flg1-flg2]+dp[1][flg2])
ans = dp[D][(1<<N)-1]
# ans -= xmean*xmean*D
print(ans)
