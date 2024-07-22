import sys
import math

N2 = int(input())

N = 10**6
# 高速素因数分解
sosuu = [0]
divs = [-1]*(N+1)
divs[1] = 1
for i in range(2, N+1):
    if divs[i] == -1:
        for j in range(1, N//i+1):
            divs[i*j] = i
        sosuu.append(i)

sosuu.append(math.inf)
def is_ok(arg, x):
    return sosuu[arg]*x<=N2


def bisect(ng, ok, x):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid, x):
            ok = mid
        else:
            ng = mid
    return ok

ans = 0
for i in range(len(sosuu)):
    if i<=1:
        continue
    cnt = bisect(i, 0, sosuu[i]**3)
    ans += cnt
    if sosuu[i]**3 > N2:
        break
print(ans)
