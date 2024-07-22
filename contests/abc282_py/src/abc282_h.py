import sys
import math

N, S = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
cumsumb = [0]
for b in B:
    cumsumb.append(cumsumb[-1]+b)

# l<=x<rかつsum(B[l:r])=Sな r最小なl, rを見つける
def findlr(x):
    pass

# sum(B[l:r])=>Sな l最小なlを見つける
def is_ok(l, r):
    return cumsumb[r]-cumsumb[l]>=S

def bisect(ng, ok, r):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid, r):
            ok = mid
        else:
            ng = mid
    return ok


for i, b
