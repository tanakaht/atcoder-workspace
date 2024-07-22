import sys
import math


T = int(input())
for caseid in range(1, T+1):
    N = int(input())-1
    def is_ok(arg):
        return 13*arg*(arg+1)<=N

    def bisect(ng, ok):
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            if is_ok(mid):
                ok = mid
            else:
                ng = mid
        return ok
    i = bisect(N, 0)
    N -= 13*i*(i+1)
    ans = chr(N//(i+1)+65)
    print(f"Case #{caseid}: {ans}")
