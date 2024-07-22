N, X = map(int, input().split())
ord('a') # 97
ord('z') # 122
ord('A') # 65
ord('Z') # 90

print(chr(65+((X-1)//N)))
