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

N = int(input())
A = list(map(int, input().split()))
uf = UnionFind(N)
g_ = [[] for _ in range(N)]
noteqs = []
i = j = 0
while i < N:
    while j < A[i] + 1 and 0<=i-j and i + j < N:
        uf.union(i-j, i+j)
        j += 1
    k = 1
    while i - k >= 0 and k + A[i - k] + 1 < j:
        k += 1
    i += k
    j -= k

for i, a in enumerate(A):
    if i-a-1>=0 and i+a+1<N:
        noteqs.append((i-a-1, i+a+1))
        g_[i-a-1].append(i+a+1)
        g_[i+a+1].append(i-a-1)

for i, j in noteqs:
    if uf.find(i)==uf.find(j):
        print('No')
        sys.exit()

gidx2idx = {gidx: i for i, gidx in enumerate(sorted(uf.roots()))}
gidx2n = {}
g = [set() for _ in range(len(gidx2idx))]
for i, j in noteqs:
    g[gidx2idx[uf.find(i)]].add(gidx2idx[uf.find(j)])
    g[gidx2idx[uf.find(j)]].add(gidx2idx[uf.find(i)])

ans = []
for i in range(N):
    gidx = uf.find(i)
    if gidx2n.get(gidx) is not None:
        ans.append(gidx2n[gidx]+1)
        continue
    appeared = set()
    for v in g[gidx2idx[gidx]]:
        if gidx2n.get(v) is not None:
            appeared.add(gidx2n[v])
    for i in range(N+1):
        if i not in appeared:
            gidx2n[gidx] = i
            ans.append(i+1)
            break

print("Yes")
print(*ans, sep=" ")
