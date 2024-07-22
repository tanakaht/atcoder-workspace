import sys
import math

N, M = map(int, input().split())

def f(n, cur):
    if n==0:
        print(*cur, end=" \n")
    else:
        for x in range(cur[-1]+1, M+1):
            cur.append(x)
            f(n-1, cur)
            cur.pop()
for x in range(1, M+1):
    f(N-1, [x])
