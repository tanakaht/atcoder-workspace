import sys
import math

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    S = input()
    S2 = S.replace("?", "1")
    cnt1 = S.count("1")
    S2 = list(S.replace("?", "1"))
    S = list(S)
    if cnt1==0:
        maxcnt = 0
        cnt = 0
        for i in range(N):
            if S2[i]=="1":
                cnt += 1
                if maxcnt==K and cnt==K:
                    maxcnt = math.inf
                maxcnt = max(maxcnt, cnt)
            else:
                cnt = 0
        if maxcnt == K:
            print("Yes")
        else:
            print("No")
    else:
        l, r = math.inf, -math.inf
        for i in range(N):
            if S[i]=="1":
                l = min(l, i)
                r = max(r, i)
        if "".join(S2[l:r+1]).count("0")!=0:
            print("No")
        else:
            if r-l+1==K:
                print("Yes")
            elif r-l+1>K:
                print("No")
            else:
                # 1同士は連結, 間は?で埋まっている
                maxcnt = 0
                hasone = False
                cnt = 0
                for i in range(N):
                    if S2[i]=="1":
                        if S[i]=="1":
                            hasone = True
                        cnt += 1
                        if hasone:
                            maxcnt = max(maxcnt, cnt)
                    else:
                        cnt = 0
                        hasone = False
                if maxcnt == K:
                    print("Yes")
                elif maxcnt>K and (l==0 or S[l-1]=="0" or r==N-1 or S[r+1]=="0"):
                    print("Yes")
                else:
                    print("No")
