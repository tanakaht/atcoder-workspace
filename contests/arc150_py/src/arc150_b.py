import sys
import math

T = int(input())
for _ in range(T):
    A, B = map(int, input().split())
    ans = math.inf
    for h in range(1, 10**5):
        if A*h>=B:
            ans = min(ans, A*h-B)
        G = B+((h-(B%h))%h)
        F = G//h
        if F>=A:
            ans = min(ans, (G-B+F-A))
    for F in range(A, 10**5):
        G = B + ((F-(B%F))%F)
        ans = min(ans, (G-B+F-A))
    print(ans)
