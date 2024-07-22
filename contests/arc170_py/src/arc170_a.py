import sys
import math

N = int(input())
S = input()
T = input()

a2b = []
b2a = []
init_a = N+1
last_b = -1
for i, (s, t) in enumerate(zip(S, T)):
    if t=='A':
        init_a = min(init_a, i)
    if t=='B':
        last_b = i
    if s == 'A' and t == 'B':
        a2b.append(i)
    elif s == 'B' and t == 'A':
        b2a.append(i)
b2a = b2a[::-1]
cnt = 0
for i in b2a:
    if i > last_b:
        print(-1)
        sys.exit()
    if len(a2b)>0 and i<a2b[-1]:
        cnt += 1
        a2b.pop()
    else:
        cnt += 1
if len(a2b)>0 and init_a>a2b[0]:
    print(-1)
    sys.exit()
cnt += len(a2b)
print(cnt)
