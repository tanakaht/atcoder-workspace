import heapq

N, L = map(int, input().split())
A = list(map(int, input().split()))
if sum(A)!=L:
    A.append(L-sum(A))
ans=0
heapq.heapify(A)
while len(A)>1:
    b1 = heapq.heappop(A)
    b2 = heapq.heappop(A)
    heapq.heappush(A, b1+b2)
    ans += b1+b2
print(ans)
