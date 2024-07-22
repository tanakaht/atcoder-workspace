import sys
import math
from collections import deque

N = int(input())
S = input()
arcs = []
for i in range(N-2):
    if S[i:i+3] == "ARC":
        cnt = 0
        j = 1
        while 0<=i-j and i+2+j<N and S[i-j]=="A" and S[i+2+j]=="C":
            cnt += 1
            j += 1
        arcs.append(cnt)
arcs = deque(sorted(arcs))
ans = 0
while arcs:
    ans += 1
    if ans%2==1:
        # 常に最大じゃないけど多分大丈夫
        cnt = arcs.pop()
        if cnt==0:
            pass
        elif cnt == 1:
            arcs.appendleft(cnt-1)
        else:
            arcs.append(cnt-1)
    else:
        arcs.popleft()
print(ans)
