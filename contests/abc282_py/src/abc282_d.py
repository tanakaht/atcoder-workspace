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
UV = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
for a, b in UV:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)

uf = UnionFind(N)
group = [None]*N
for a in range(N):
    for b in g[a]:
        if uf.find(a)==uf.find(b):
            print(0)
            sys.exit()
    if g[a]:
        b0 = g[a][0]
        for b1 in g[a][1:]:
            uf.union(b0, b1)

ans = N*(N-1) - 2*M
for g in uf.all_group_members().values():
    ans -= len(g)*(len(g)-1)
print(ans//2)
