import sys
import math

N, X = map(int, input().split())
A = sorted(list(map(int, input().split())))
LR = [[A[-1], A[-1]]]
for i in range(N-1):
    l, r = A[i], A[i]
    prel, prer = l, r
    while r<A[-1]:
        prel, prer = l, r
        l, r = 2*l, 2*r+X
    if l<=A[-1]<=r:
        pass
    else:
        LR.append([prer, l])
ans = math.inf
rmax = A[-1]
for l, r in sorted(LR):
    ans = min(ans, rmax-l)
    rmax = max(rmax, r)
for _ in range(60):
    ans = min(max(0, 2*ans-X), ans)
print(ans)
