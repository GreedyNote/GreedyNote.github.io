---
title:  "[데이터 수집] ZIP파일"
excerpt: "zip 파일 포맷의 파일을 압축해제하자"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/python_logo.jpg

categories:
  - data preprocessing
tags:
  - 라벨인코딩
---
# Categorical Encoding 의 두가지 방법
일반적으로 우리가 가진 데이터셋에는 범주형 데이터가 포함되어 있다. 하지만 기계는 숫자만 이해한다. 그래서 우리는 이러한 범주형 데이터를 숫자로 변환해주는 과정이 필요한 것이다. 머신러닝 알고리즘을 동작하기 위해 이러한 전처리 과정을 Categorical Encoding이라고 한다.

그림 1

범주형 인코딩을 하기위해서 어떻게 하면될까? 대표적으로 두가지 방법이 있다
- **Label Encoding**
- **One-Hot Encoding**  

오늘 이 포스팅에선 Label Encoding에 대해 다룰 것이다.

## label Encoding 이란?
라벨 인코딩이란 알파벳 오더순으로 숫자를 할당해주는 것을 말한다.

그림2

위의 Country 열의 데이터를 숫자로 바꾸면 다음과 같아진다

그림3

하지만 라벨인코딩은 적용에 난점이 있다. 위의 데이터에서 Country라는 데이터는 순서나 랭크가 없다.  
그러나 라벨인코딩을 수행하면 결국엔 알파벳 순으로 랭크가 되는 것이고, 그로 인해서 랭크된 숫자정보가  
모델에 잘못 반영될 수가 있다. 그래서 나온것이 또다른 방법인 One-Hot-Encoding 이다.

## Label Encoding 실습
이제부터 실제로 라벨 인코딩을 해보겠습니다. sklearn의 LabelEncoder 클래스를 사용합니다.


```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
```

실제 데이터를 생성합니다. X_test에만 `Korea`가 있습니다


```python
X_train = np.array(['India', 'USA', 'Japan', 'USA'])
X_test = np.array(['India', 'Korea', 'USA', 'Japan'])
```

라벨 인코더 생성


```python
encoder = LabelEncoder()
```

X_train 데이터를 이용해 피팅하고 라벨숫자로 변환한다


```python
encoder.fit(X_train)
encoder.transform(X_train)
```




    array([0, 2, 1, 2], dtype=int64)



X_test 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가해야한다


```python
for label in np.unique(X_test):
    if label not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, label)
encoder.transform(X_test)
```




    array([0, 2, 4, 1], dtype=int64)



만약 이런식으로 X_test에만 존재하는 신규 클래스를 추가하지않고 바로 transform을 하면 Value Error 발생

그림 4

반대로 변환하는 것도 가능합니다. 인덱스를 입력하면 본래 값을 반환합니다.


```python
encoder.inverse_transform([0,1,2,3])
```




    array(['India', 'Japan', 'USA', 'Korea'], dtype='<U5')
