from collections import Counter

c = Counter(map(int, input().split()))
if sorted(c.values()) == [2, 3]:
    print("Yes")
else:
    print("No")
