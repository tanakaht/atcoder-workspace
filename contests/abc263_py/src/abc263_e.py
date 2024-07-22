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

N = int(input())
A = list(map(int, input().split()))
MOD = 998244353
def op(l, r):
    ul, dl = l
    ur, dr = r
    return (ul*dr+ur*dl)%MOD, (dl*dr)%MOD
st = SegmentTree(N, segfunc=op, ele=(0, 1))
for i in range(N-2, -1, -1):
    l = i+1
    r = min(N, i+A[i]+1)
    u, d = st.query(l, r)
    u = (u+(A[i]+1)*d)%MOD
    d = (d*A[i])%MOD
    # e = (A[i]+1) + st.query(l, r)
    # e = (e*pow(A[i], MOD-2, MOD))%MOD
    st.update(i, (u, d))
u, d = st.get_val(0)
print((u*pow(d, MOD-2, MOD))%MOD)
