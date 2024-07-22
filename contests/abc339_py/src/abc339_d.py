import sys
import math
from collections import defaultdict, deque

N = int(input())
S = [input() for _ in range(N)]
init_p = []
for i in range(N):
    for j in range(N):
        if S[i][j] == 'P':
            init_p.append((i, j))
s = init_p[0][0]*N*N*N + init_p[0][1]*N*N + init_p[1][0]*N + init_p[1][1]
appered = set([s])
q = deque([(s, 0)])
while q:
    p, d = q.popleft()
    i1, j1, i2, j2 = p//(N*N*N), (p%(N*N*N))//(N*N), (p%(N*N))//N, p%N
    if i1==i2 and j1==j2:
        print(d)
        sys.exit()
    for (i1_, j1_, i2_, j2_) in [(i1+1, j1, i2+1, j2), (i1-1, j1, i2-1, j2), (i1, j1+1, i2, j2+1), (i1, j1-1, i2, j2-1)]:
        if not (0 <= i1_ < N and 0 <= j1_ < N and S[i1_][j1_]!="#"):
            i1_, j1_ = i1, j1
        if not (0 <= i2_ < N and 0 <= j2_ < N and S[i2_][j2_]!="#"):
            i2_, j2_ = i2, j2
        p_ = i1_*N*N*N+j1_*N*N+i2_*N+j2_
        if p_ not in appered:
            appered.add(p_)
            q.append((p_, d+1))
print(-1)
