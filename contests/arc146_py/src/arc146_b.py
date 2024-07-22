import sys
import math

N, M, K = map(int, input().split())
A = list(map(int, input().split()))

def cost(a, x):
    maxbit = None
    # x
    for i in range(31):
        xbit = (x>>i)&1
        abit = (a>>i)&1
        if xbit and (not abit):
            maxbit = i
    if maxbit is None:
        return 0
    a = a&((1<<(maxbit+1))-1)
    x = x&((1<<(maxbit+1))-1)
    return max(0, x-a)

x = 0
for i in range(30, -1, -1):
    x = x+(1<<i)
    costs = sorted([cost(a, x) for a in A])
    if sum(costs[:K])<=M:
        pass
    else:
        x = x-(1<<i)
print(x)
