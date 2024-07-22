import sys
import math

Q = int(input())
Qs = [list(map(int, input().split())) for _ in range(Q)]
for q in Qs:
    if q[0]==0:
        x = q[1]
        pass
    elif q[0]==1:
        x = q[1]
        k = q[2]
        pass
    elif q[0]==2:
        x = q[1]
        k = q[2]
        pass
