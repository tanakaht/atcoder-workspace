import sys
import math

N, M = map(int, input().split())
P = list(map(int, input().split()))
XY = [list(map(int, input().split())) for _ in range(M)]

hoken = [0]*N
for x, y in XY:
    x -= 1
    hoken[x] = max(hoken[x], y+1)


children = [[] for _ in range(N)]
for i, p in enumerate(P):
    i += 1
    p -= 1
    children[p].append(i)

q = [0]
while q:
    u = q.pop()
    for v in children[u]:
        hoken[v] = max(hoken[v], hoken[u]-1)
        q.append(v)

cnt = 0
for x in hoken:
    cnt += x>0
print(cnt)
