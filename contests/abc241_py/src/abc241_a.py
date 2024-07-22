import sys
import math

A = list(map(int, input().split()))
ans = 0
for _ in range(3):
    ans = A[ans]
print(ans)
