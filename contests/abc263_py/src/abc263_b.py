import sys
import math

N = int(input())
P = [0] + list(map(int, input().split()))
def solve(i):
    if i==1:
        return 0
    return 1+solve(P[i-1])
print(solve(N))
