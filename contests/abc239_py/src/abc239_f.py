import sys
import math

class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def all_group_members(self):
        d = {root: [] for root in self.roots()}
        for i in range(self.n):
            d[self.find(i)].append(i)
        return d

    def __str__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())


N, M = map(int, input().split())
D = list(map(int, input().split()))
AB = [list(map(int, input().split())) for _ in range(M)]
if sum(D)!=2*N-2:
    print(-1)
    sys.exit(0)

uf = UnionFind(N)
for a, b in AB:
    a -= 1
    b -= 1
    uf.union(a, b)
    D[a] -= 1
    D[b] -= 1
nodes_d = {k: [] for k in uf.roots()}
for idx, mem in uf.all_group_members().items():
    for i in mem:
        for _ in range(D[i]):
            nodes_d[idx].append(i)
nodes_d = sorted(nodes_d.values(), key=len)[::-1]
nodes = nodes_d[0]
anss = []
for nodes_ in nodes_d[1:]:
    if len(nodes_)==0 or len(nodes)==0:
        print(-1)
        sys.exit(0)
    anss.append((nodes_.pop(), nodes.pop()))
    nodes.extend(nodes_)
while len(nodes)>=2:
    anss.append((nodes.pop(), nodes.pop()))
if len(nodes)!=0:
    print(-1)
    sys.exit(0)
for u, v in anss:
    print(u+1, v+1)
