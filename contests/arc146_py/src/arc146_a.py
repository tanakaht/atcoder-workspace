import sys
import math

N = int(input())
A = sorted(list(map(int, input().split())))[::-1]
A = list(map(str, A))
ans = max(
    int(A[0]+A[1]+A[2]),
    int(A[0]+A[2]+A[1]),
    int(A[1]+A[0]+A[2]),
    int(A[1]+A[2]+A[0]),
    int(A[2]+A[1]+A[0]),
    int(A[2]+A[0]+A[1]),
)
print(ans)
