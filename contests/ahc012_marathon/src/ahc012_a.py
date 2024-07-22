import sys
import math
import random
from time import time
from collections import defaultdict
import bisect

TS = time()
TIMELIMIT = 2.500
input = sys.stdin.readline
N, K = map(int, input().split())
A = list(map(int, input().split()))
XY = sorted([list(map(int, input().split())) for _ in range(N)])
XY_sortedx = sorted(XY, key=lambda x: x[0])
XY_sortedy = sorted(XY, key=lambda x: x[1])
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


def get_x(b):
    return (b, -10000000, b+1, 10000000)

def get_y(b):
    return (-10000000, b, 10000000, b+1)


Nx = math.ceil(math.sqrt(sum(A)))+5
# Ny = math.ceil(math.sqrt(sum(A)))+5
Ny = 10 # math.ceil(math.sqrt(sum(A)))+5
xs = []
ys = []
for i in range(Nx+1):
    b = -10000 + (i*20000)//Nx
    xs.append(b)
for i in range(Ny+1):
    b = -10000 + (i*20000)//Ny
    ys.append(b)
printans(xs, ys)

def cnts2score(cnts):
    if cnts[0]>K:
        return -1
    ret = 0
    for i in range(1, 11):
        if A[i-1] < cnts[i]:
            continue
        ret += (i*4-1)*(A[i-1]-cnts[i])*(A[i-1]-cnts[i]+1)/A[i-1]
    return 1000000-ret
    score = 10000000-sum([(A[i]-cnts[i+1])*(A[i]-cnts[i+1])*i/A[i] for i in range(10)])# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    score -= 110000*cnts[11]
    return score
    score = sum([min(A[i], cnts[i+1])*i for i in range(10)])# -sum([0.1*max(0, d2[i+1]-A[i]) for i in range(10)])
    score -= 11*cnts[11]
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

def dp_xs(ys):
    dp = [(-1, [0]*12 , -1, 1) for _ in range(N+1)] # i-1番目のいちごで線を引くときの、(最高スコア, counter, 直前の線の位置, 引いた線の数)
    cumsums = [[0]*len(ys)] # i-1番目のいちごで線を引くときのcnt
    cur_cnt = [0]*len(ys)
    for i, (x, y) in enumerate(XY_sortedx):
        yi = bisect.bisect_left(ys, y)-1
        cur_cnt[yi] += 1
        cumsums.append([c for c in cur_cnt])
        for j in range(max(0, i-100), i+1):
            pre_cnt = cumsums[j]
            pre_counter = [c for c in dp[j][1]]
            for c in [c1-c2 for c1, c2 in zip(cur_cnt, pre_cnt)]:
                pre_counter[min(c, 11)] += 1
            score = cnts2score(pre_counter)-(dp[j][3]>=98-len(ys))*1000000000000000000000000000000
            if dp[i+1][0] < score:
                dp[i+1] = (score, pre_counter, j-1, dp[j][3]+1)
    score = dp[-1][0]
    pre_i = dp[-1][2]
    xs = []
    while pre_i >= 0:
        xs.append(XY_sortedx[pre_i][0])
        pre_i = dp[pre_i][2]
    xs.append(-150000)
    return xs[::-1], score

def dp_ys(xs):
    dp = [(-1, [0]*12 , -1, 1) for _ in range(N+1)] # i-1番目のいちごで線を引くときの、(最高スコア, counter, 直前の線の位置, 引いた線の数)
    cumsums = [[0]*len(xs)] # i-1番目のいちごで線を引くときのcnt
    cur_cnt = [0]*len(xs)
    for i, (x, y) in enumerate(XY_sortedy):
        xi = bisect.bisect_left(xs, x)-1
        cur_cnt[xi] += 1
        cumsums.append([c for c in cur_cnt])
        for j in range(max(0, i-100), i+1):
            pre_cnt = cumsums[j]
            pre_counter = [c for c in dp[j][1]]
            for c in [c1-c2 for c1, c2 in zip(cur_cnt, pre_cnt)]:
                pre_counter[min(c, 11)] += 1
            score = cnts2score(pre_counter)-(dp[j][3]>=98-len(xs))*1000000000000000000000000000000
            if dp[i+1][0] < score:
                dp[i+1] = (score, pre_counter, j-1, dp[j][3]+1)
    score = dp[-1][0]
    pre_i = dp[-1][2]
    ys = []
    while pre_i >= 0:
        ys.append(XY_sortedy[pre_i][1])
        pre_i = dp[pre_i][2]
    ys.append(-150000)
    return ys[::-1], score


best_score = 0
best_ans = [[], []]
loop_cnt = 0
while time()-TS<TIMELIMIT:
    loop_cnt += 1
    xs, score = dp_xs(ys)
    if score>best_score:
        best_score = score
        best_ans = [[x for x in xs], [y for y in ys]]
    printans(xs, ys)
    ys, score = dp_ys(xs)
    if score>best_score:
        best_score = score
        best_ans = [[x for x in xs], [y for y in ys]]
    printans(xs, ys)


# print(loop_cnt)
printans(best_ans[0], best_ans[1])
