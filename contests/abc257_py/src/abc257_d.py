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
xyP = [list(map(int, input().split())) for _ in range(N)]


ans = math.inf
def is_ok(s):
    g = [[] for _ in range(N)]
    for i in range(N):
        xi, yi, Pi = xyP[i]
        for j in range(N):
            xj, yj, Pj = xyP[j]
            if abs(xi-xj)+abs(yi-yj)<=Pi*s:
                g[i].append(j)
    for start in range(N):
        q = [start]
        appeared = [False]*N
        appeared[start] = True
        while q:
            u = q.pop()
            for v in g[u]:
                if not appeared[v]:
                    q.append(v)
                    appeared[v] = True
        if sum(appeared)==N:
            return True
    return False

def bisect(ng, ok):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok

print(bisect(0, 10**9*5))
