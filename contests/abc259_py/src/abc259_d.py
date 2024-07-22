import sys
import math

N = int(input())
sx, sy, tx, ty = map(int, input().split())
XYR = [list(map(int, input().split())) for _ in range(N)]
g = [[] for _ in range(N)]
for i, (x, y, r) in enumerate(XYR):
    for j, (x_, y_, r_) in enumerate(XYR):
        d = (x-x_)**2+(y-y_)**2
        if (r-r_)**2<=d<=(r+r_)**2:
            g[i].append(j)
for i, (x, y, r) in enumerate(XYR):
    if (x-sx)**2+(y-sy)**2==r*r:
        si = i
        break
for i, (x, y, r) in enumerate(XYR):
    if (x-tx)**2+(y-ty)**2==r*r:
        ti = i
        break

appeared = [False]*N
appeared[si] = True
q = [si]
while q:
    u = q.pop()
    if u==ti:
        print("Yes")
        sys.exit()
    for v in g[u]:
        if not appeared[v]:
            q.append(v)
            appeared[v] = True
print("No")
