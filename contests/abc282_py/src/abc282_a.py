ord('a') # 97
ord('z') # 122
ord('A') # 65
ord('Z') # 90
print("".join([chr(i) for i in range(65, 91)][:int(input())]))
