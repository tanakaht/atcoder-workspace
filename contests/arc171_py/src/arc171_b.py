import sys
import math

N = int(input())
A = list(map(lambda x: int(x)-1, input().split()))
MOD = 998244353
for i, a in enumerate(A):
    if a<i:
        print(0)
        sys.exit()

used = [None]*N
valid = set()
for i, a in enumerate(A):
    if i!=a:
        if used[a] is None:
            used[a] = i
            valid.add(i)
        if used[i] is not None:
            print(0)
            sys.exit()
available = 0
ans = 1
for i, a in enumerate(A):
    if i==a:
        if used[i] is None:
            available += 1
        ans = (ans*available)%MOD
        available -= 1
    else:
        if i in valid:
            available += 1
print(ans)
