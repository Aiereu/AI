# 파이썬을 연습한 코드입니다.
# 주의! Import가 순서 상 도중에 삽입되어 있습니다.
# output
print("hello world!")
print(1)
print([1, 2, 3])
print(1, 2, 3)
print('python', 'hello')
print("pthon", "hello")

# Input

a = input()
print(a)
age = input("당신의 나이는 ? : ")
# 입력 후 출력
print(age)

# 자료형, int, float, boolean
mint5 = 11
print(mint5)

f = 3.14
print(f)
print(type(f))

print("aaa", 'bbb', mint5, f)

print(True)
print(False)

# 자료형 리스트, 튜플, 딕트

student = ['안녕', '하이', '헬로우']
print(student)

for stu in student:
    print(stu)
student.append("hello")

import random

print(random.choice(student))

student[0] = "안녕 하세요"
print(student)

mtuple = ('hello', 'laa')
print(mtuple)
# mtu[0] = "11" 튜플은 수정이 불가

dic = {'man': '남', "안녕": '인사', "인사": '안녕'}
print(dic['인사'])
# 자료형 리스트[], 튜플()(수정불가), 딕트{}(키워드-값)

# 형변환

mint = "안녕하세요"
print(type(str(mint)))
print(list(mint))
# myint 를 리스트 형태로 변환하여 출력한다 형변환

# 문자열

mstr0 = "반가워요"
print(type(mstr0))
mstr0 = '안녕하세요'
print(type(mstr0))

mstr0 = """안녕
구텐탁
헬로우
반가워
오하이요"""

print(mstr0)
# 문자열 완료

# 문자열 포맷팅

mstr = 'my name is %s %s' % ('dong geon', 'hello')

print(mstr)

mstr1 = 'my name is {}'.format('헬로우')

print(mstr1)

# 포맷팅 중괄호에 순서 값을 넣으면 그 순서 값이 출력된다.
mstr2 = '{1} x {0} = {2}'.format(2, 3, 2 * 3)
print(mstr2)

# 문자열 인덱싱

mstr3 = "헬로우 안녕하세요"
print(mstr3[2])
print(mstr3[-2])
# 문자열 인덱싱 마이너스는 -1 부터 뒤로 셈

# 문자열 슬라이싱

print(mstr3[4:7])
# 4이상 ~ 7미만
print(mstr3[4:])
# 4이상
print(mstr3[:3])
# 3미만

# 문자열 메소드

print(mstr3.split())

lang = '한국어 영어 독일어 불어 스페인어 아랍어 러시아어'
print(lang.split())
l1 = lang.split('어', 3)
# 값을 넣지 않으면 공백 기준, 첫번째 파라미터는 자르는 기준, 두번째는 자르는 횟수
print(l1)

# 독스트링
""" 이것도 주석입니다 """
# """로 주석처리, 함수 설명할때 사용

# end, 이스케이프 코드
print('안녕', '헬로우', '잘있어')
print('안녕', end='/')
# 넣으면 마지막에 엔터 대신 넣어짐

print('안녕\t그래\n잘있어')
# 이스케이프 \t 탭 \n 개행

# LIST
# 이뮤터블 : 값을 변경할 수 없는 것 ex) 튜플, 뮤터블 : 있는 것 ex) 리스트

mlist1 = []
print(type(mlist1))
mlist1 = [1, 2, 3]
print(mlist1)
mlist1.append('찹살떡')
print(mlist1)
mlist1.insert(0, 'hello')
print(mlist1)
mlist1.pop(0)
print(mlist1)
# 팝은 프린트에 넣으면 사라지는 인덱스 값이 출력됨, 삽입이랑 추가는 넣으면 none 뜸
mlist1.pop()
print(mlist1)

# 리스트 인덱싱, 슬라이싱

del mlist1[0]  # 삭제

print(mlist1)

del mlist1[0]
del mlist1[0]
mlist1.append('hello')
mlist1.append('잘있어')
mlist1.append('한국어')
mlist1.append('하이')
print(mlist1[0:3])

# 리스트 메서드

mlist1.sort()
print(mlist1)
# 문자 혹은 숫자 한가지 종류로만 이루어진 리스트만 가능

mlist1.append('hello')
print(mlist1.count('hello'))

mlist1.pop()
print(len(mlist1))

print(mstr3)
print(len(mstr3))

# 튜플 이뮤터블

mtuple1 = 1, 2, 3
# 패킹
print(mtuple1)
print(type(mtuple1))

# 패킹, 언패킹

n1, n2, n3 = mtuple1
# 언패킹
print(n1, n2, n3)

n1, n2 = n2, n1
# 오른쪽 패킹 & 왼쪽 언패킹
print(n1, n2)

# for 규칙에 의해 도는 반복문

for i in lang:
    print(i)

# range 시작 수 이상 종료 수 미만의 숫자 범위를 갖는 배열

for i in range(0, 100):
    print(i)

# 이중 for 문

for i in range(1, 10):
    for j in range(1, 10):
        print('{}x{}={}'.format(i, j, i * j))

# 컨프리헨션

num = range(1, 10)
odd_n = []

for i in num:
    if i % 2 == 1:
        odd_n.append(i)
        print(i)

n = [i for i in num if i % 2 == 0]  # 출력 값 for문 if문 순으로 구성
print(n)

# 할당 연산자

# += -= *= /= 를 말함

# 산술 연산자

# + - * / 를 말함 * 에스퍼리스크

# %로 홀짝 구분하기

n = range(0, 8)

for i in n:
    if not i % 2 != 1:
        print(i)
    else:
        print("짝수 : {}".format(i))

# 문자열 연산자
mstr4 = '안녕하셈'
mstr5 = ' 그랭 '
print(mstr4 + ' ' + mstr5)

print(mstr5 * 10)


def cls():
    print('\n' * 100)


# 비교 연산자

# == > < <= >=

# 논리 연산자

# and or not

print(True and False)
print(not True)
print(not False)

h = 160
age = 20

print(h < 180 and age > 10)

# 멤버쉽 연산자
print('멤버쉽 연산자')
print('한국어' in lang)
print('영어' not in lang)

# if

name = '사람'

if name == '사람':
    print(" HI ", name)

# else, elif

if name == '사람':
    print(" 안녕 ! ", name)
elif name == '내친구':
    print(" 안녕 ! ", name)
elif name == '친구의친구':
    print(" 안녕 ! ", name)
else:
    print("누구세요?")

# while 반복문

c = 0

while c < 2:
    print("이건 앞의 반복문", c)
    c += 1

# continue break

c = 0

while c < 11:
    c += 1
    if c < 4:
        continue
    print(c)
    if c == 7:
        break

# dictionary

mdict = {}
mdict['hello'] = '안녕'
mdict['hi'] = '안녕하'
mdict['오하이요'] = ' 좋은 아침?'
print(mdict)
mdict[3] = '3번째'
print(mdict)

del mdict[3]

print(mdict)

# dictionary method

for i in mdict.values():
    print("값", i)
for i in mdict.keys():
    print("키", i)
for i in mdict.items():
    print("아이템", i)
for i in mdict:
    print(i)


# 함수 , 반환값이 여러개인 경우 튜플을 던져줌

def add(n1, n2):
    return n1 + n2, n1 * n2


a, b = add(3, 4)
print(a)
print(b)

# 모듈

# 여러 함수를 묶어둔 것이 모듈 import를 통해 모듈을 추가
#  모듈은 나중에 import해도 작동한다. 다만 변수 혹은 배열은 사용 이전에 정의되어야 한다.

# 랜덤

a = ['안녕', 'hi', '오하이요', '구텐탁']

print(random.choice(a))

import random

print(random.choice(a))

print(random.sample(a, 3))

print(random.sample(range(1, 46), 6))

print(random.randint(3, 10))

# 객체

# 함수와 데이터를 모아서 객체를 만들 수 있다.

import datetime

cal = datetime.date.today()

print(cal)
