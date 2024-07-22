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
HWD = sorted([sorted(list(map(int, input().split()))) for _ in range(N)])
appered = set([0])
for h, w, d in HWD:
    # appered.add(h)
    appered.add(w)
    # appered.add(d)
id2length = sorted(appered)
length2id = {l: i for i, l in enumerate(id2length)}
st = SegmentTree(len(id2length))
q = []
for h, w, d in HWD:
    if q and q[0][0]<h:
        for x,y,z in q:
            st.update(length2id[y], min(st.get_val(length2id[y]), z))
        q = []
    d_min = st.query(0, length2id[w])
    if d>d_min:
        print("Yes")
        sys.exit()
    q.append([h, w, d])

print("No")
