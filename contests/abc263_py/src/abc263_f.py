import sys
import math

N = int(input())
C = [[0]+list(map(int, input().split())) for _ in range(pow(2, N))]

dp = [[0]*pow(2, N) for _ in range(N+1)]

for i in range(N):
    w = pow(2, i)
    for l in range(0, pow(2, N), w*2):
        rmax = max([dp[i][j] for j in range(l+w, l+2*w)])
        lmax = max([dp[i][j] for j in range(l, l+w)])
        for j in range(l, l+w):
            dp[i+1][j] = dp[i][j] + C[j][i+1]-C[j][i] + rmax
        for j in range(l+w, l+2*w):
            dp[i+1][j] = dp[i][j] + C[j][i+1]-C[j][i] + lmax
print(max(dp[-1]))
