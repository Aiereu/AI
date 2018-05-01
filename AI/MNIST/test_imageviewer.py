import scipy.misc
import numpy

# 원하는 이미지 파일을 등록 후 실행
print("searching images...")
img_array = scipy.misc.imread('./query-images/test_three.png', flatten=True)
print("img_array 불러온 값 \n", img_array)
img_data = 255.0 - img_array.reshape(784)
print("img_data reshape 후 \n", img_data)

# 정규화
img_data = (img_data / 255.0 * 0.99) + 0.01

print("img_data 정규화 후 \n", img_data)

# 최댓값, 최솟값
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))
