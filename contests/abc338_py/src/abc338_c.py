import sys
import math

N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

ans = 0
for x in range(max(Q)+1):
    tmpans = math.inf
    for q, a, b in zip(Q, A, B):
        if a*x>q:
            tmpans = -math.inf
            break
        if b==0:
            continue
        tmpans = min(tmpans, (q-a*x)//b)
    ans = max(ans, tmpans+x)
print(ans)
