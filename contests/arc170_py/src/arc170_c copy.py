import sys
import math

N, M = map(int, input().split())
S = list(map(int, input().split()))
MOD = 998244353
idxs = []
for i, s in enumerate(S):
    if s == 1:
        idxs.append(i)
idxs = idxs[::-1]
# idxs[i]=jの時の右側が決定済み
dp = [[0]*5001 for _ in range(len(idxs))]
xx = [[0]*100 for _ in range(100)]
for i in range(len(idxs)-1, idxs[0]+1):
    dp[0][i] = pow(M, N-idxs[0]-1, MOD)
for i in range(1, len(idxs)):
    for j in range(len(idxs)-1-i, idxs[i]+1):
        for j2 in range(j+1, j+idxs[i-1]-idxs[i]):
            dp[i][j] = (dp[i][j] + dp[i-1][j2]*xx[idxs[i-1]-idxs[i]-1][j2-j-1])%MOD
ans = 0
for j in range(5001):
    ans = (ans + dp[-1][j]*xx[idxs[-1]][j])%MOD
