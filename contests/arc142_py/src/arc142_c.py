import sys
import math

N = int(input())
ans = math.inf
mins = []
for i in range(3, N+1):
    print("?", 1, i)
    x = int(input())
    print("?", 2, i)
    y = int(input())
    if ans>x+y:
        ans = x+y
        mins = [i]
    elif ans==x+y:
        mins.append(i)
if ans!=3:
    print("!", ans)
    sys.exit()
else:
    if len(mins)<2:
        print("!", 1)
    else:
        print("?", mins[0], mins[1])
        x = int(input())
        if x==1:
            print("!", 3)
        else:
            print("!", 1)
