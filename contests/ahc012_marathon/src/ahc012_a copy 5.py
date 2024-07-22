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


def is_upper(p1p2, p):
    """p1-p2の直線よりpが上にいるか
    """
    x, y = p
    x1, y1, x2, y2 = p1p2
    if x1==x2:
        return x1>=x
    y_ = (y1*(x2-x)+y2*(x-x1))/(x2-x1)
    return y_<=y


Nx = math.ceil(math.sqrt(sum(A)))+5
Ny = math.ceil(math.sqrt(sum(A)))+5
xs = []
ys = []
for i in range(Nx+1):
    b = -10000 + (i*20000)//Nx
    xs.append(b)
for i in range(Nx+1):
    b = -10000 + (i*20000)//Ny
    ys.append(b)
printans(xs, ys)

grids = [[[] for _ in range(Ny)] for _ in range(Nx)]
for x, y in XY:
    xi = bisect.bisect_left(xs, x)-1
    yi = bisect.bisect_left(ys, y)-1
    grids[xi][yi].append((x, y))

cnts = {i: 0 for i in range(-1, 11)}
def addcnts(cnts, cnt):
    if cnt>10:
        cnts[-1] += 1
    else:
        cnts[cnt] += 1

def deccnts(cnts, cnt):
    if cnt>10:
        cnts[-1] -= 1
    else:
        cnts[cnt] -= 1

for xi in range(Nx):
    for yi in range(Ny):
        cnt = len(grids[xi][yi])
        addcnts(cnts, cnt)

def get_score(cnts):
    score = 1000000*sum([min(A[i], cnts[i+1]) for i in range(10)])/sum(A)# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    cost = 0
    X = {i: x for i, x in enumerate(A)}
    Y = {i-1: cnts[i] for i in range(1, 11)}
    for i in range(10):
        for j in range(10):
            cnt = min(X[i], Y[j])
            cost += abs(i-j)*cnt
            X[i] -= cnt
            Y[j] -= cnt
            if not X[i]:
                break
        cost += X[i]*(11-i)
    return score-1000*(TIMELIMIT-(time()-TS))*cost

cur_score = get_score(cnts)
loop_cnt = 0
C = TIMELIMIT*10000
while time()-TS<TIMELIMIT:
    loop_cnt += 1
    # print(loop_cnt, time()-TS)
    if random.random()>0.5:
        xi = randint(1, Nx)
        b_pre = xs[xi]
        b_new = randint(max(xs[xi]-100, xs[xi-1]), min(xs[xi]+100, xs[xi+1]))
        for yi in range(Ny):
            deccnts(cnts, len(grids[xi][yi]))
            deccnts(cnts, len(grids[xi-1][yi]))
            tmpxy = grids[xi-1][yi]+grids[xi][yi]
            grids[xi-1][yi] = []
            grids[xi][yi] = []
            for x, y in tmpxy:
                grids[xi-1+(x>b_new)][yi].append((x, y))
            addcnts(cnts, len(grids[xi][yi]))
            addcnts(cnts, len(grids[xi-1][yi]))
        new_score = get_score(cnts)
        forceLine = (TIMELIMIT-(time()-TS)) / C;
        # if cur_score<=new_score:
        if (cur_score <= new_score) or (forceLine >random.random()):
            cur_score = new_score
            xs[xi] = b_new
            # print(cur_score)
            printans(xs, ys)
        else:
            deccnts(cnts, len(grids[xi][yi]))
            deccnts(cnts, len(grids[xi-1][yi]))
            tmpxy = grids[xi-1][yi]+grids[xi][yi]
            grids[xi-1][yi] = []
            grids[xi][yi] = []
            for x, y in tmpxy:
                grids[xi-1+(x>b_pre)][yi].append((x, y))
            addcnts(cnts, len(grids[xi][yi]))
            addcnts(cnts, len(grids[xi-1][yi]))
    else:
        yi = randint(1, Ny)
        b_pre = ys[yi]
        b_new = randint(max(ys[yi]-100, ys[yi-1]), min(ys[yi]+100, ys[yi+1]))
        for xi in range(Nx):
            deccnts(cnts, len(grids[xi][yi]))
            deccnts(cnts, len(grids[xi][yi-1]))
            tmpxy = grids[xi][yi-1]+grids[xi][yi]
            grids[xi][yi-1] = []
            grids[xi][yi] = []
            for x, y in tmpxy:
                grids[xi][yi-1+(y>b_new)].append((x, y))
            addcnts(cnts, len(grids[xi][yi]))
            addcnts(cnts, len(grids[xi][yi-1]))
        new_score = get_score(cnts)
        forceLine = (TIMELIMIT-(time()-TS)) / C;
        # if cur_score<=new_score:
        if (cur_score <= new_score) or (forceLine >random.random()):
            cur_score = new_score
            ys[yi] = b_new
            # print(cur_score)
            printans(xs, ys)
        else:
            deccnts(cnts, len(grids[xi][yi]))
            deccnts(cnts, len(grids[xi][yi-1]))
            tmpxy = grids[xi][yi-1]+grids[xi][yi]
            grids[xi][yi-1] = []
            grids[xi][yi] = []
            for x, y in tmpxy:
                grids[xi][yi-1+(y>b_pre)].append((x, y))
            addcnts(cnts, len(grids[xi][yi]))
            addcnts(cnts, len(grids[xi][yi-1]))
# print(loop_cnt)
# print(cur_score, 1000000*sum([min(A[i], cnts[i+1]) for i in range(10)])/sum(A))
