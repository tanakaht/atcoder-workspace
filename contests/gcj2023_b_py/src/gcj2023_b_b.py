import sys
import math
def extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        gcd, x, y = extended_gcd(b, a % b)
        return (gcd, y, x - (a // b) * y)

def find_x(d, y, N):
    gcd, a, b = extended_gcd(d, N)
    if y % gcd != 0:
        return None
    else:
        x = (a * (y // gcd)) % N
        return x

T = int(input())
for caseid in range(1, T+1):
    W, N, D = map(int, input().split())
    X = list(map(int, input().split()))
    ans = 0
    for i in range(W//2):
        diff = (X[i]-X[-i-1]+N)%N
        if diff==0:
            continue
        x = find_x(D, diff, N)
        x2 = find_x(D, N-diff, N)
        if x is None and x2 is None:
            ans = "IMPOSSIBLE"
            break
        elif x is None:
            ans += x2
        elif x2 is None:
            ans += x
        else:
            ans += min(x,x2)
    print(f"Case #{caseid}: {ans}")
