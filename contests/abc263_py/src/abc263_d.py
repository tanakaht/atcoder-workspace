import sys
import math

N, L, R = map(int, input().split())
A = list(map(int, input().split()))

sumA = sum(A)
rdiff = [0]*(N+1)
ldiff = [0]*(N+1)
cur = 0
curmin = 0
for i in range(N):
    cur += L - A[i]
    curmin = min(curmin, cur)
    ldiff[i+1] = curmin

cur = 0
curmin = 0
for i in range(N, 0, -1):
    cur += R - A[i-1]
    curmin = min(curmin, cur)
    rdiff[i] = curmin

ans = sumA
for i in range(N+1):
    ans = min(ans, sumA+ldiff[i]+rdiff[i])
print(ans)
