# 직접 피보나치를 구현해보았습니다.


def Pibo(s1, s2):
    print("시작 순서 ", s1, s2)
    count = 3
    n1 = 1
    n2 = 1
    n3 = 0
    if s1 == 1 or s1 == 2:
        print("1")
    if s1 == 1:
        print("1")
    for i in range(s2 + 1):
        n3 = n1 + n2
        if count >= s1 and count <= s2:
            print(n3)
        count += 1
        n2 = n1
        n1 = n3


a = int(input("시작할 번째를 입력하십시오 : "))
b = int(input("마지막 번째를 입력하십시오. 최소 시작할 번째와의 간격은 3이상입니다 : "))
Pibo(a, b)
