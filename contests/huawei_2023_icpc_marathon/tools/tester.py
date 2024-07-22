import argparse
import sys
import numpy as np
import math

def read_input():
    with open(sys.argv[1]) as f:
        eles = f.read().split()
        n = int(eles[0])
        X = list(map(lambda x: np.float128(x), eles[1:]))
    return n, X

def eval_d(v1, v2):
    if abs(v1+v2)>np.finfo(np.float64).max:
        return np.float128("inf")
    return np.float64(v1)+np.float64(v2)

def eval_s(v1, v2):
    if abs(v1+v2)>np.finfo(np.float32).max:
        return np.float128("inf")
    return np.float32(v1)+np.float32(v2)

def eval_h(v1, v2):
    if abs(v1+v2)>np.finfo(np.float16).max:
        return np.float128("inf")
    return np.float16(v1)+np.float16(v2)


class Siki:
    def __init__(self, eles, flg: str):
        self.eles = eles
        self.flg = flg

    def eval(self, X: list):
        if self.flg == 'n':
            return (X[self.eles[0]-1], 0)
        res = self.eles[0].eval(X)
        val = res[0]
        c_cost = res[1]
        for ele in self.eles[1:]:
            res = ele.eval(X)
            c_cost += res[1]
            if self.flg == 'd':
                c_cost += 4
                val = eval_d(val, res[0])
            elif self.flg == 's':
                c_cost += 2
                val = eval_s(val, res[0])
            elif self.flg == 'h':
                c_cost += 1
                val = eval_h(val, res[0])
        return (val, c_cost)

    def order(self):
        if self.flg == 'n':
            return [self.eles[0]]
        ret = []
        for ele in self.eles:
            for v in ele.order():
                ret.append(v)
        return ret

    def __str__(self) -> str:
        return f"{self.flg}{','.join(map(str, self.eles))}"

def read_output():
    with open(sys.argv[2]) as f:
        ret = []
        line = f.readline().rstrip()
        line = line.replace("}", ",}")
        for ele_ in line.split(","):
            for ele in ele_.split(":"):
                if ele.startswith("{"):
                    ret.append(ele[1])
                elif ele == "}":
                    ret.append("}")
                else:
                    ret.append(int(ele))
    return parse(ret)

def parse(ss: list):
    eles = []
    flg = ss[0]
    idx = 1
    cnt = 0
    tmpss = []
    while idx < len(ss)-1:
        if isinstance(ss[idx], int):
            if cnt==0:
                eles.append(Siki(eles=[ss[idx]], flg="n"))
            else:
                tmpss.append(ss[idx])
        elif ss[idx] == "}":
            cnt -= 1
            tmpss.append(ss[idx])
            if cnt == 0:
                eles.append(parse(tmpss))
                tmpss = []
        else:
            cnt += 1
            tmpss.append(ss[idx])
        idx += 1
    return Siki(eles, flg)



def test():
    n, X = read_input()
    siki = read_output()
    v, c_cost = siki.eval(X)
    real_v = 0
    for ele in sorted(X, key=lambda x: abs(x)):
        real_v += ele
    real_v = np.float64(real_v)
    if v==np.float128("inf"):
        score = 0
        return score
    A = pow(max(abs(real_v-v)/max(abs(real_v), pow(10, -200)), pow(10, -20)), 0.05)
    W = c_cost
    idxs = siki.order()
    cnt = 0
    P = 0
    for i in range(len(idxs)//16):
        for p in idxs[i*16:(i+1)*16]:
            if abs(p-idxs[i*16])>15:
                cnt += 1
                P += cnt/20000
    C = (W+P)/(n-1)
    D = 10/math.sqrt(C+0.5)
    print(abs(real_v-v), real_v, P, W)
    return D/A

score = test()
print(f"{score}")
