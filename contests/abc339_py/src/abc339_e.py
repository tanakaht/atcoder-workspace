import sys
import math
import bisect

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

N, D = map(int, input().split())
A = list(map(int, input().split()))
last_match = [[-1]*2 for _ in range(N)]
A_sorted = sorted(A)
v2idx = {v: i for i, v in enumerate(A_sorted)}
st = SegmentTree(N, max, 0)
for i, a in enumerate(A):
    l = bisect.bisect_left(A_sorted, a-D)
    r = bisect.bisect_right(A_sorted, a+D)
    v = st.query(l, r)+1
    st.update(v2idx[a], v)
print(st.query(0, N))
