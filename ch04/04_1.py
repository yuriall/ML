import pandas as pd
fish = pd.read_csv('fish.csv')
# head() = 처음 행 5개 출력
print(fish.head())
# unique() = 어떤 종류의 생선이 있는지 열에서 고유 값 추출
print(pd.unique(fish['Species']))

#생선 이름 -> target, 나머지 열 선택하여 새로운 데이터
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])

# 타깃 데이터 만들기
fish_target = fish['Species'].to_numpy()

# train, test set 분류
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# train, test set 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#K최근접 이웃의 3개로 설정 후 train, test set 결과 보기
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# sklearn에서의 타깃값은 알파벳 순서 (csv 파일에 있는 순서와 다름)
print(kn.classes_)

# 테스트 세트에 있는 상위 5개의 데이터로 결과 보기
print(kn.predict(test_scaled[:5]))

# 테스트 세트에 있는 상위 5개의 데이터 확률 출력하기
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) # 소수점 4번째 자리까지만 표기하세요

# 출력한 확률이 맞는지 확인하기 위해 test set에서 4번째 데이터를 활용
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
# 단점 발생 : 이웃을 3으로 지정해놓아서 확률이 1/3, 2/3 과 같은 식만 나옴

# 로지스틱 함수 이용하여 도미와 빙어 행 해보기
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
#확률 출력
print(lr.predict_proba(train_bream_smelt[:5]))
#계수 확인
print(lr.coef_, lr.intercept_)
# 처음 샘플 5개의 z 값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
# 해당 값들을 시그모이드 함수에 대입
from scipy.special import expit
print(expit(decisions))


#이제 생선 7개에 대해서 해보자!
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# 샘플 5개에 대한 예측
print(lr.predict(test_scaled[:5]))
# 샘플 5개에 대한 확률 가져오기
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=5))

#방정식 계수 모양 출력 -> 다중분류는 클래스 마다 z 값을 하나씩 계산함
print(lr.coef_.shape, lr.intercept_.shape)
# 이진 분류 -> 시그모이드 함수 / 다중 분류 -> 소프트맥스 함수

# z1~z7 까지의 값을 구하고 소프트맥스 함수를 사용하여 확률로 바꾸자
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))