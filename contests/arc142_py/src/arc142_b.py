import sys
import math

N = int(input())
ans = [[None]*N for _ in range(N)]
cur = 1
for i in range(0, N, 2):
    for j in range(N):
        ans[i][j] = cur
        cur += 1
for i in range(1, N, 2):
    for j in range(N):
        ans[i][j] = cur
        cur += 1
for x in ans:
    print(*x)
