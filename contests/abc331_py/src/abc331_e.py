import sys
import math
import heapq

N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
CD = set([tuple(map(lambda x: int(x)-1, input().split())) for _ in range(L)])

A_i = sorted([(v, i) for i, v in enumerate(A)])
B_i = sorted([(v, i) for i, v in enumerate(B)])
q = [(N-1, M-1)]
appeared = set([(N-1, M-1)])
while q:
    n, m = heapq.heappop(q)
    a, ia = A_i[n]
    b, ib = B_i[m]
    if (ia, ib) not in CD:
        print(a+b)
        sys.exit()
    if n > 0 and (n-1, m) not in appeared:
        heapq.heappush(q, (n-1, m))
        appeared.add((n-1, m))
    if m > 0 and (n, m-1) not in appeared:
        heapq.heappush(q, (n, m-1))
        appeared.add((n, m-1))
