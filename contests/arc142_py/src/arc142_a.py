import sys
import math

N, K = map(int, input().split())
if K%10==0:
    print(0)
    sys.exit()
if K>N:
    print(0)
    sys.exit()
revK = int(str(K)[::-1])
if K>revK:
    print(0)
    sys.exit()


ans = 0
x = K
while x<=N:
    ans += 1
    x *= 10
if K!=revK:
    x = revK
    while x<=N:
        ans += 1
        x *= 10
print(ans)
