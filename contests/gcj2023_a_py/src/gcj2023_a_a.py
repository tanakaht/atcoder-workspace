import sys
import math

T = int(input())
for caseid in range(1, T+1):
    A = input().split()
    N = int(input())
    appeared = set()
    flg = True
    for _ in range(N):
        w = "".join(map(lambda x: A[ord(x)-65], input()))
        if w in appeared:
            flg = False
        appeared.add(w)
    if not flg:
        ans = "YES"
    else:
        ans = "NO"
    print(f"Case #{caseid}: {ans}")
