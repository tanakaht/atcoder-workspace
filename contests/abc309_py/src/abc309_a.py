a, b = map(int, input().split())

if a in [3,6,9]:
    print("No")
elif b-a==1:
    print("Yes")
else:
    print("No")
