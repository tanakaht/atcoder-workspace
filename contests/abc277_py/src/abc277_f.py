import sys
import math

H, W = map(int, input().split())
As = [list(map(int, input().split())) for _ in range(H)]
A_min_max = []
for A in As:
    a_max = max(A)
    if a_max==0:
        continue
    a_min = min([a for a in A if a!=0])
    A_min_max.append((a_min, a_max))
pre_amax = 0
for a_min, a_max in sorted(A_min_max):
    if a_min<pre_amax:
        print("No")
        sys.exit()
    pre_amax = a_max

g = [set() for _ in range(W+H*W+2)]
g_rev = [set() for _ in range(W+H*W+2)]

for A in As:
    for i, a in sorted(enumerate(A),key=lambda x: x[1]):
        if a==0:
            continue
        g[i].add(W+a)
        g_rev[W+a].add(i)
        g[W+a+1].add(i)
        g_rev[i].add(W+a+1)
for a in range(H*W+1):
    g[W+a+1].add(W+a)
    g_rev[W+a].add(W+a+1)
print(g)
# toposo
q = [i for i in range(W+H*W+2) if len(g[i])==0]
while q:
    i = q.pop()
    for j in g_rev[i]:
        g[j].remove(i)
        if len(g[j])==0:
            q.append(j)
print(g)
if sum([len(g[i]) for i in range(W+H*W+2)])==0:
    print("Yes")
else:
    print("No")
