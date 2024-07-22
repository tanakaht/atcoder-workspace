import sys
import math

N = int(input())
A = [list(map(int, input())) for _ in range(N)]
B = [[y for y in x] for x in A]
B[0] = [A[1][0]]+B[0][:-1]
B[-1] = B[-1][1:] + [A[-2][-1]]
for i in range(1, N-1):
    B[i][0] = A[i+1][0]
    B[i][-1] = A[i-1][-1]
for b in B:
    print("".join(map(str, b)))
