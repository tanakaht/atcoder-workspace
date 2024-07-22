from re import A
import sys

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

with open(sys.argv[1]) as f:
    N, K, L = map(int, f.readline().split())
    AB = [list(map(int, f.readline().split())) for _ in range(K)]
    C = [list(map(lambda x: int(x)-1, f.readline().split())) for _ in range(N)]

with open(sys.argv[2]) as f:
    ans = [int(f.readline())-1 for _ in range(K)]

id2xy = {}
rest = set()
for x in range(N):
    for y in range(N):
        if C[x][y] != -1:
            id2xy[C[x][y]-1] = (x, y)
        else:
            rest.add((x, y))
# 連結か判定
uf = UnionFind(N*N+1)
for x in range(N):
    for y in range(N):
        if (x, y) in rest:
            uf.union(x*N+y, N*N)
            continue
        for x_, y_ in [(x+1, y), (x, y+1)]:
            if 0<=x_<N and 0<=y_<N and (x_, y_) not in rest:
                if ans[C[x][y]]==ans[C[x_][y_]]:
                    uf.union(x*N+y, x_*N+y_)

isjointed = (len(uf.roots())==L+1)

ps, qs = [0]*L, [0]*L
for k in range(K):
    l = ans[k]
    a, b = AB[k]
    ps[l] += a
    qs[l] += b

base = 10**6 if isjointed else 10**3
score = round(base * min(min(ps)/max(ps), min(qs)/max(qs)))
print(score)
