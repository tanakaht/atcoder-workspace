import sys
import math

def cnt(l, r):
    return max(0, r-l+1)

def base(x):
    return 10**(len(str(x))-1)

def cnt2(r):
    return cnt(base(r), r)

def is_ok(arg, R):
    return min(arg*10, arg+base(arg)*10)>R

def bisect(ng, ok, R):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid, R):
            ok = mid
        else:
            ng = mid
    return ok

T = int(input())
for _ in range(T):
    L, R = map(int, input().split())
    L_ = max(bisect(-1, R, R), L)
    ans = cnt(L_, R)
    print(ans)
