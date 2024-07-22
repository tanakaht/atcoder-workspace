import sys
import math

N = int(input())
A = list(map(int, input().split()))
ans = N
pre = 0
for x in A[::-1]:
    pre += x
    if pre >3:
        break
    ans -= 1
print(ans)
