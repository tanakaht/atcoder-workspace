# 全部log(N)
class Mex:
    def op(self, a, b):
        if a[0]<b[0] and a[1]==0:
            return a
        return b

    def __init__(self, N, A=None):
        self.st = SegmentTree(N, ele=[math.inf, 1], segfunc=self.op)
        self.N = N
        if A:
            self.st.init(A)
        else:
            self.st.init([[i, 0] for i in range(N)])

    def add(self, x):
        _, cnt = self.st.get_val(x)
        self.st.update(x, [x, cnt+1])

    def remove(self, x):
        _, cnt = self.st.get_val(x)
        if cnt >= 1:
            self.st.update(x, [x, cnt-1])
        else:
            self.st.update(x, [x, 0])

    def mex(self):
        return self.st.query(0, self.N)[0]
