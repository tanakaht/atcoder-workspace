import sys
import math

N, M = map(int, input().split())
S = list(map(int, input().split()))
MOD = 998244353
# [0, i)まで決定済み、mex(A[0..i]))がj
dp = [[0]*N for _ in range(N+1)]
dp[0][0] = 1
for i in range(1, N+1):
    for j in range(i+1):
