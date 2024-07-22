import collections
import math

N, M = map(int, input().split())
ans = [-1 for _ in range(N*N)]
ans[0] = 0

Q = collections.deque()
Q.append([0, 0, 0])

g = []
for k in range(int(math.sqrt(M))+1):
    l = int(math.sqrt(M-k*k))
    if k*k+l*l==M:
        g.append((k, l))
        g.append((k, -l))
        g.append((-k, l))
        g.append((-k, -l))

for i in range(N):
    for j in range(N):
        for k, l in g:
            x = i*i+j*j+k+l
            x = x*x
raise ValueError

while Q:
    d, i, j = Q.popleft()
    for k, l in g:
        if 0<=i+k<N and 0<=j+l<N:
            if ans[(i+k)*N+(j+l)]==-1:
                d_ = d+1
                Q.append([d_, i+k, j+l])
                ans[(i+k)*N+(j+l)] = d_

for i in range(N):
    print(*ans[i*N:(i+1)*N])
