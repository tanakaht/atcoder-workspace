import sys
import math

N = int(input())
XY = [list(map(int, input().split())) for _ in range(N)]

# 　面積の2倍
def cal_area(i, j, k):
    i %= N
    j %= N
    k %= N
    xi, yi = XY[i]
    xj, yj = XY[j]
    xk, yk = XY[k]
    xj -= xi
    yj -= yi
    xk -= xi
    yk -= yi
    return abs(xj*yk-xk*yj)

a8 = 0
for i in range(2, N):
    a8 += cal_area(0, i-1, i)

area = 0
i, j = 0, 1
ans = math.inf
for i in range(N):
    while area*4<a8:
        area += cal_area(i, j, j+1)
        j += 1
        ans = min(ans, abs(4*area-a8))
    area -= cal_area(i, i+1, j)
    ans = min(ans, abs(4*area-a8))
print(ans)
