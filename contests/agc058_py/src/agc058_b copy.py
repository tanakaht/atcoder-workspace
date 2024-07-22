import sys
import math

sys.setrecursionlimit(100000)
class SegmentTree:
    def __init__(self, n, segfunc=min, ele=10**10):
        self.ide_ele = ele
        self.num = pow(2, (n-1).bit_length())
        self.seg = [self.ide_ele] * (2 * self.num)
        self.segfunc = segfunc

    def init(self, init_val):
        #set_val
        for i in range(len(init_val)):
            self.seg[i+self.num-1] = init_val[i]
        #built
        for i in range(self.num-2, -1, -1):
            self.seg[i] = self.segfunc(self.seg[2*i+1], self.seg[2*i+2])

    def update(self, k, x):
        k += self.num-1
        self.seg[k] = x
        while k:
            k = (k-1)//2
            self.seg[k] = self.segfunc(self.seg[k*2+1], self.seg[k*2+2])

    # [p, q)のop
    def query(self, p, q):
        if q <= p:
            return self.ide_ele
        p += self.num-1
        q += self.num-2
        res = self.ide_ele
        while q-p > 1:
            if p & 1 == 0:
                res = self.segfunc(res, self.seg[p])
            if q & 1 == 1:
                res = self.segfunc(res, self.seg[q])
                q -= 1
            p = p//2
            q = (q-1)//2
        if p == q:
            res = self.segfunc(res, self.seg[p])
        else:
            res = self.segfunc(self.segfunc(res, self.seg[p]), self.seg[q])
        return res

    def get_val(self, k):
        k += self.num-1
        return self.seg[k]

N = int(input())
P = list(map(int, input().split()))
MOD = 998244353
def segfunc(a, b):
    if a[1]>b[1]:
        return a
    else:
        return b
st_max = SegmentTree(N, segfunc, ele=(-1, -1))
st_max.init(list(enumerate(P)))
lr_d = {}
for i in range(N):
    l, r = i, i
    while l>=0 and P[l]<=P[i]:
        l -= 1
    while r<N and P[r]<=P[i]:
        r += 1
    lr_d[i] = (l, r)

# [l, r)
dp = [[None]*(N+1) for _ in range(N+1)]
for i in range(N+1):
    dp[i][i] = 1
def solve(l, r):
    if dp[l][r] is not None:
        return dp[l][r]
    maxi, v = st_max.query(l, r)
    ret = 0
    for l_ in range(l, maxi+1):
        for r_ in range(maxi+1, r+1):
            ret = (ret+solve(l, l_)*solve(r_, r))%MOD
    dp[l][r] = ret
    return ret
solve_ord = []
dp_dummy = [[None]*N for _ in range(N+1)]
def solve_dummy(l, r):
    if dp[l][r] is not None:
        return dp[l][r]
    maxi, v = st_max.query(l, r)
    ret = 0
    for l_ in range(l, maxi):
        for r_ in range(r, maxi):
            ret += 1
    dp[l][r] = ret
    return ret

for i, p in sorted(enumerate(P)):
    l, r = lr_d[i]
    solve(l, r)


print(solve(0, N))
for i, x in enumerate(dp):
    print(i, *x)
