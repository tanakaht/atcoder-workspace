import sys
import math

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

N, Q = map(int, input().split())
A = list(map(int, input().split()))
Qs = [list(map(int, input().split())) for _ in range(Q)]
MOD = 998244353
def op(l, r):
    return ((l[0]+r[0])%MOD, (l[1]+r[1])%MOD, (l[2]+r[2])%MOD)
st = SegmentTree(N, op, ele=(0, 0, 0))
inv2 = pow(2, MOD-2, MOD)
init_vs = []
for i in range(N):
    init_vs.append((A[i], ((3-2*i)*A[i])%MOD, ((i-1)*(i-2)*A[i])%MOD))
st.init(init_vs)
for query in Qs:
    if query[0] == 1:
        _, x, v = query
        x -= 1
        A[x] = v
        st.update(x, (A[x], ((3-2*x)*A[x])%MOD, ((x-1)*(x-2)*A[x])%MOD))
    if query[0] == 2:
        _, x = query
        x -= 1
        k0, k1, k2 = st.query(0, x+1)
        ans = (k0*x*x+k1*x+k2)%MOD
        ans = (ans*inv2)%MOD
        print(ans)
