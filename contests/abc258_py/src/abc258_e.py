import sys
import math

N, Q, X = map(int, input().split())
W = list(map(int, input().split()))
Wsum = sum(W)
g = [-1]*N
Wcumsum = [0]
for w in W:
    Wcumsum.append(Wcumsum[-1]+w)
for w in W:
    Wcumsum.append(Wcumsum[-1]+w)
for w in W:
    Wcumsum.append(Wcumsum[-1]+w)
def is_ok(to, fr):
    w_ = (Wcumsum[to]-Wcumsum[fr])
    return w_<(X%Wsum)

def bisect(ng, ok, fr):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid, fr):
            ok = mid
        else:
            ng = mid
    return ok


for fr in range(N):
    cnt = X//Wsum*N
    # ちょい怖い
    if X%Wsum==0:
        g[fr] = (fr, cnt)
        continue
    to = (bisect(fr+N+1, fr, fr)+1)
    g[fr] = (to%N, cnt+(to-fr))
    cnt -= 1
    w -= W[fr]
g2 = [[to for to, cnt in g]]
for _ in range(42):
    g_ = g2[-1]
    g2.append([g_[g_[i]] for i in range(N)])

for _ in range(Q):
    K = int(input())-1
    i = 0
    for flg, g_ in zip(bin(K)[2:][::-1], g2):
        if flg=="1":
            i = g_[i]
    print(g[i][1])
