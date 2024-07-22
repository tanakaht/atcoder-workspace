import sys
import math
import random
from time import time
from collections import defaultdict
import bisect

TS = time()
TIMELIMIT = 2.5
input = sys.stdin.readline
N, K = map(int, input().split())
A = list(map(int, input().split()))
XY = sorted([list(map(int, input().split())) for _ in range(N)])
anss = []
def printans(xs, ys):
    print(len(xs)+len(ys))
    for b in xs:
        print(*get_x(b))
    for b in ys:
        print(*get_y(b))

def randint(a, b):
    """[a,b)の整数
    """
    return a+int(random.random()*(b-a))


def get_random_points():
    b = int(random.random()*30000-15000)
    if random.random()<0.5:
        return (0, b, 1, b)
    else:
        return (b, 0, b, 1)
    r = random.uniform(7000, 20000)
    theta = random.uniform(0, 2*math.pi)
    return int(r*math.cos(theta)), int(r*math.sin(theta))

def get_x(b):
    return (b, -10000000, b+1, 10000000)

def get_y(b):
    return (-10000000, b, 10000000, b+1)


Nx = math.ceil(math.sqrt(sum(A)))+5
Ny = math.ceil(math.sqrt(sum(A)))+5
xs = []
ys = []
for i in range(Nx+1):
    b = -11000 + (i*22000)//Nx
    xs.append(b)
for i in range(Nx+1):
    b = -11000 + (i*22000)//Ny
    ys.append(b)
printans(xs, ys)

def get_grids(xs, ys):
    grids = [[[] for _ in range(Ny)] for _ in range(Nx)]
    for x, y in XY:
        xi = bisect.bisect_left(xs, x)-1
        yi = bisect.bisect_left(ys, y)-1
        grids[xi][yi].append((x, y))
    return grids

def grids2cnts(grids):
    cnts = {i: 0 for i in range(-1, 11)}
    for xi in range(Nx):
        for yi in range(Ny):
            cnt = len(grids[xi][yi])
            if cnt>10:
                cnts[-1] += 1
            else:
                cnts[cnt] += 1
    return cnts

def cnts2score(cnts):
    score = sum([min(A[i], cnts[i+1]) for i in range(10)])# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    return score
    j_pre, cnt_pre = 1, 0
    for i in range(1, 11):
        a = A[i-1]
        for j in range(j_pre, 11):
            cnt = min(cnts[j]-cnt_pre*(j_pre==j), a)
            score -= abs(i-j)*cnt
            a -= cnt
            if cnt<cnts[j]-cnt_pre*(j_pre==j):
                break
        score -= a*(100-i)
        j_pre, cnt_pre = j, cnt
    return score

def get_score(xs, ys):
    return cnts2score(grids2cnts(get_grids(xs, ys)))

cur_score = get_score(xs, ys)
loop_cnt = 0
C = TIMELIMIT*10000
while time()-TS<TIMELIMIT:
    loop_cnt += 1
    # print(loop_cnt, time()-TS)
    if random.random()>0.5:
        xi = randint(1, Nx)
        b_pre = xs.pop(xi)
        b_new = randint(-11000, 11000)
        xi_new = bisect.bisect_left(xs, b_new)
        xs.insert(xi_new, b_new)
        new_score = get_score(xs, ys)
        forceLine = (TIMELIMIT-(time()-TS)) / C;
        if cur_score<=new_score:
        #if (cur_score <= new_score) or (forceLine >random.random()):
            cur_score = new_score
            # print(cur_score)
            printans(xs, ys)
        else:
            xs.pop(xi_new)
            xs.insert(xi, b_pre)
            xs = sorted(xs)
    else:
        yi = randint(1, Ny)
        b_pre = ys.pop(yi)
        b_new = randint(-11000, 11000)
        yi_new = bisect.bisect_left(ys, b_new)
        ys.insert(yi_new, b_new)
        new_score = get_score(xs, ys)
        forceLine = (TIMELIMIT-(time()-TS)) / C;
        if cur_score<=new_score:
        #if (cur_score <= new_score) or (forceLine >random.random()):
            cur_score = new_score
            # print(cur_score)
            printans(xs, ys)
        else:
            ys.pop(yi_new)
            ys.insert(yi, b_pre)
            ys = sorted(ys)
# print(loop_cnt)
