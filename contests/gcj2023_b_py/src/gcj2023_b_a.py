import sys
import math

T = int(input())
for caseid in range(1, T+1):
    N = int(input())
    A = list(map(int, input().split()))
    La, Ra, Lb, Rb = map(lambda x: int(x)-1, input().split())
    Asum = sum(A)
    A_cumsum = [0]
    for n in A:
        A_cumsum.append(A_cumsum[-1]+n)
    ans = 0
    def pos2bob(xa, xb):
        if xa<xb:
            return Asum-A_cumsum[(xa+xb)//2+1]
        else:
            return A_cumsum[(xa+xb-1)//2+1]
    for xa in range(La, Ra+1):
        bmax = 0
        kouho = [Lb, Rb]
        if Lb<=xa-1<=Rb:
            kouho.append(xa-1)
        if Lb<=xa+1<=Rb:
            kouho.append(xa+1)
        for xb in kouho:
            bmax = max(bmax, pos2bob(xa, xb))
        ans = max(ans, Asum-bmax)
    print(f"Case #{caseid}: {ans}")
