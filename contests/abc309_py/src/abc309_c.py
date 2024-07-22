import sys
import math

N, K = map(int, input().split())
AB = sorted([list(map(int, input().split())) for _ in range(N)])
cnt = 0
for a, b in AB:
    cnt += b
if cnt<=K:
    print(1)
    sys.exit()
for a, b in AB:
    cnt -= b
    if cnt <= K:
        print(a+1)
        sys.exit()
