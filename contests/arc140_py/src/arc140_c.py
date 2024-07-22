import sys
import math

N, X = map(int, input().split())
if N==2:
    print(*[X, X%2+1])
    sys.exit()


ans1 = []
u, d = N-(N==X), 1+(1==X)
is_u = 1
for _ in range(N-1):
    if is_u:
        ans1.append(u)
        u -= 1
        u -= u==X
    else:
        ans1.append(d)
        d += 1
        d += d==X
    is_u = is_u ^ 1
ans1.append(X)
if abs(ans1[-1]-ans1[-2])==1 and abs(ans1[-2]-ans1[-3])==2:
    print(*ans1[::-1])
    sys.exit()
ans2 = []
u, d = N-(N==X), 1+(1==X)
is_u = 0
for _ in range(N-1):
    if is_u:
        ans2.append(u)
        u -= 1
        u -= u==X
    else:
        ans2.append(d)
        d += 1
        d += d==X
    is_u = is_u ^ 1
ans2.append(X)
print(*ans2[::-1])
