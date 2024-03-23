# 로지스틱 회귀 방정식의 y 값 (=z) 가 0~1 사이의 확률로 표현되어야 함
# 큰 음수 -> 0, 큰 양수 -> 1 로 바꾸는 방법
# 시그모이드 (=로지스틱) 함수 사용

# 함수 그려보기
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 로지스틱 회귀로 이진 분류 수행하기
# 0.5보다 크면 양성, 0.5 보다 작으면 음성
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])