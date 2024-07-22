import sys
import math

def get_nearest_2(v, X):
    if v<X[0]:
        return [X[0], X[1]]
    elif v>X[-1]:
        return [X[-1], X[-2]]
    l = 0
    r = len(X)-1
    while r-l>1:
        m = (l+r)//2
        if X[m]<v:
            l = m
        else:
            r = m
    vals = [X[l], X[r]]
    if l>0:
        vals.append(X[l-1])
    if r<len(X)-1:
        vals.append(X[r+1])
    vals = sorted(vals, key=lambda x: abs(x-v))
    return vals[:2]
T = int(input())
for _ in range(T):
    N = int(input())
    A = sorted(list(map(int, input().split())))
    B = sorted(list(map(int, input().split())))
    found = False
    mindist = 10**9
    for a in A[::-1]:
        # b最小での判定
        if a>B[0]:
            v = get_nearest_2(a, A)[1]
            if a-B[0]<v<a+B[0]:
                continue
        # a<bでの判定
        if mindist<a:
            found = True
            break
    if found:
        print("Alice")
    else:
        print("Bob")
