import sys
import math
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())
XY = [list(map(int, input().split())) for _ in range(Q)]
x2yi = defaultdict(list)
for i, (x, y) in enumerate(XY):
    x2yi[x-1].append((y-1, i))
anss = ["No"]*Q
nsqrt = int(math.sqrt(N))
bi, prebi = 0, math.inf
appered_A = set()
appered_B = set()
AuB = set()
for ai, a in enumerate(A):
    appered_A.add(a)
    if a in appered_B:
        AuB.add(a)
        for y, i in x2yi[ai]:
            if len(appered_A)==len(AuB)==len(appered_B) and prebi<=y<bi:
                anss[i] = "Yes"
            else:
                anss[i] = "No"
    else:
        while bi<N and B[bi]!=a:
            appered_B.add(B[bi])
            if B[bi] in appered_A:
                AuB.add(B[bi])
            bi += 1
        if bi==N:
            bi, prebi = math.inf, math.inf
            break
        prebi = bi
        appered_B.add(B[bi])
        AuB.add(B[bi])
        while bi<N and B[bi] in appered_B:
            bi += 1
        for y, i in x2yi[ai]:
            if len(appered_A)==len(AuB)==len(appered_B) and prebi<=y<bi:
                anss[i] = "Yes"
            else:
                anss[i] = "No"

print(*anss, sep='\n')
