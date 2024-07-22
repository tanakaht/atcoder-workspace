import sys
import math

def dist(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    return abs(p1x-p2x)+abs(p1y-p2y)

T = int(input())
for _ in range(T):
    B, K, Sx, Sy, Gx, Gy = map(int, input().split())
    ans = dist((Sx, Sy), (Gx, Gy))*K
    def under(x):
        return (x//B)*B
    def over(x):
        return ((x//B)*B)+B
    Ss = [(under(Sx), Sy), (over(Sx), Sy), (Sx, under(Sy)), (Sx, over(Sy))]
    Gs = [(under(Gx), Gy), (over(Gx), Gy), (Gx, under(Gy)), (Gx, over(Gy))]
    S_s = [(under(Sx),under(Sy)), (over(Sx), under(Sy)), (under(Sx), over(Sy)), (over(Sx), over(Sy))]
    G_s = [(under(Gx),under(Gy)), (over(Gx), under(Gy)), (under(Gx), over(Gy)), (over(Gx), over(Gy))]
    kouho1 = []
    for S_ in S_s:
        dist1 = math.inf
        for S in Ss:
            dist1 = min(dist1, dist((Sx, Sy), S)*K + dist(S, S_))
        kouho1.append((dist1, S_))
    kouho2 = []
    for G_ in G_s:
        dist2 = math.inf
        for G in Gs:
            dist2 = min(dist2, dist((Gx, Gy), G)*K + dist(G, G_))
        kouho2.append((dist2, G_))
    for dist1, S_ in kouho1:
        for dist2, G_ in kouho2:
            ans = min(ans, dist1+dist2+dist(S_, G_))
    for S in Ss:
        for G in Gs:
            if (S[0]==G[0] and S[0]%B==0) or (S[1]==G[1] and S[1]%B==0):
                ans = min(ans, dist((Sx, Sy), S)*K + dist(S, G)+dist(G, (Gx, Gy))*K)
    print(ans)
