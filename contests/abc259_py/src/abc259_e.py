import sys
import math
from collections import defaultdict

N = int(input())
d = defaultdict(list)
for i in range(N):
    m = int(input())
    pe = [list(map(int, input().split())) for _ in range(m)]
    for p, e in pe:
        d[p].append((e, i))
flgs = [False]*N
for vs in d.values():
    vs = sorted(vs)
    if len(vs)==1:
        flgs[vs[0][1]] = True
    else:
        if vs[-1][0]!=vs[-2][0]:
            flgs[vs[-1][1]] = True
print(sum(flgs)+(sum(flgs)!=N))
