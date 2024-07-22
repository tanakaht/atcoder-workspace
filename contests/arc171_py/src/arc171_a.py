import sys
import math

T = int(input())
for _ in range(T):
    N, A, B = map(int, input().split())
    if A>N:
        print("No")
    else:
        haba = N-A
        tate = min(N-A, math.ceil(N/2))
        if haba*tate>=B:
            print("Yes")
        else:
            print("No")
