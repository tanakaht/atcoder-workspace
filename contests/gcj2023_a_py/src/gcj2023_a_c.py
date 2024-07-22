import sys
import math

T = int(input())
for caseid in range(1, T+1):
    N = map(int, input().split())
    S = list(map(int, input().split()))
    ans = []
    appeared = set()
    flg = True
    for s in S:
        if s in appeared:
            if s == ans[-1]:
                continue
            else:
                flg = False
                break
        else:
            ans.append(s)
            appeared.add(s)
    if flg:
        print(f"Case #{caseid}: {' '.join(map(str, ans))}")
    else:
        print(f"Case #{caseid}: IMPOSSIBLE")
