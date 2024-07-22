import sys
import math

N = int(input())
A = list(map(int, input().split()))
A_cumsum = [0]
for a in A:
    A_cumsum.append(A_cumsum[-1]+a)
print(A_cumsum[-1]-min(A_cumsum))
