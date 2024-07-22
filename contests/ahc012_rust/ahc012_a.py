import sys
import math
import random
from time import time
from collections import defaultdict

TS = time()
TIMELIMIT = 25
input = sys.stdin.readline
N, K = map(int, input().split())
A = list(map(int, input().split()))
XY = [list(map(int, input().split())) for _ in range(N)]
anss = []
def printans(anss):
    print(len(anss))
    for x in anss:
        print(*x)

def randint(a, b):
    """[a,b)の整数
    """
    return a+int(random.random()*(b-a))


def get_random_point():
    r = random.uniform(7000, 20000)
    theta = random.uniform(0, 2*math.pi)
    return int(r*math.cos(theta)), int(r*math.sin(theta))

def is_upper(p1p2, p):
    """p1-p2の直線よりpが上にいるか
    """
    x, y = p
    x1, y1, x2, y2 = p1p2
    if x1==x2:
        return x1>=x
    y_ = (y1*(x2-x)+y2*(x-x1))/(x2-x1)
    return y_<=y


for _ in range(K):
    p1 = get_random_point()
    p2 = get_random_point()
    anss.append((p1[0], p1[1], p2[0], p2[1]))
printans(anss)

state = [0]*N
for i in range(N):
    for j in range(K):
        if is_upper(anss[j], XY[i]):
            state[i] += (1<<j)

def get_score(state):
    d = defaultdict(int)
    for x in state:
        d[x] += 1
    d2 = {i: 0 for i in range(11)}
    rest = 0
    for v in d.values():
        if v<=10:
            d2[v] += 1
        else:
            rest += v
    score = sum([min(A[i], d2[i+1]) for i in range(10)])# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    # score += sum([abs(A[i]-d2[i+1])*0.1 for i in range(10)])
    # score = -sum([(A[i]-d2[i+1])*(A[i]-d2[i+1]) for i in range(10)]) - 0.5*rest*rest# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    # score = -sum([abs(A[i]-d2[i+1]) for i in range(10)]) - 0.5*rest*rest
    return score

cur_score = get_score(state)
loop_cnt = 0
C = TIMELIMIT*10000
while time()-TS<TIMELIMIT:
    loop_cnt += 1
    # print(loop_cnt, time()-TS)
    j = randint(0, K)
    pre = anss[j]
    p1 = get_random_point()
    p2 = get_random_point()
    new = (p1[0], p1[1], p2[0], p2[1])
    newstate = [0]*N
    for i in range(N):
        if is_upper(pre, XY[i])!=is_upper(new, XY[i]):
            newstate[i] = state[i]^(1<<j)
        else:
            newstate[i] = state[i]
    new_score = get_score(newstate)
    forceLine = (TIMELIMIT-(time()-TS)) / C;
    if (cur_score <= new_score) or (forceLine >random.random()):
    # if cur_score<=new_score:
        cur_score = new_score
        anss[j] = new
        state = newstate
        printans(anss)
