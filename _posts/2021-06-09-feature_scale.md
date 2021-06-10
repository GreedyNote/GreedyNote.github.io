---
title:  "[데이터 전처리] 피처 스케일링"
excerpt: "피처 스케일링 해보자"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/python_logo.jpg

categories:
  - data preprocessing
tags:
  - feature scaling
---
# 피처 스케일링 (feature scaling)
피처 스케일링은 서로 다른 변수(feature)의 값 범위를 일정한 수준으로 맞추는 작업입니다.
- **표준화 (standardize)**
- **정규화 (Normalize)**

## 스케일링을 하는 이유
행렬의 조건수(conditional number)은 가장 큰 고유치와 가장 작은 고유치의 비율을 뜻합니다. 공분산행렬 $XX^T$의 조건수가 크면  
회귀분석을 사용한 예측값도 오차가 커집니다.

$\text{condition number} = \frac{\lambda_{max}}{\lambda_{min}}$

조건수가 커지는 경우는 크게 두가지가 있습니다.
1. 변수들의 단위 차이로 인해 숫자들의 스케일이 크게 달라지는 경우 $\rightarrow$ 스케일링으로 해결
2. 다중 공선성, 즉 상관관계가 큰 독립변수들이 있는 경우 $\rightarrow$ 차원축소로 해결

![1](/assets/1_m1sm5h339.png)

## 표준화 (Standardization)
표준화는 서로 다른 범위의 변수들을 평균이 0, 표준편차가 1인 정규분포를 따르게끔 분포시키는 방법입니다.  
진폭의 감소로 각 데이터간의 간격이 감소하게 됩니다. (10000단위에서 0.1단위로 감소)

$$z_i = \frac{x_i - \bar{x}}{s}$$

### StandardScaler


```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
scaler = StandardScaler()

scaler.fit(iris.data)
iris_scaled = scaler.transform(iris.data)
iris_scaled[:5]
```




    array([[-0.90068117,  1.01900435, -1.34022653, -1.3154443 ],
           [-1.14301691, -0.13197948, -1.34022653, -1.3154443 ],
           [-1.38535265,  0.32841405, -1.39706395, -1.3154443 ],
           [-1.50652052,  0.09821729, -1.2833891 , -1.3154443 ],
           [-1.02184904,  1.24920112, -1.34022653, -1.3154443 ]])



## 정규화 (Normalization)
데이터를 특정 구간으로 바꾸는 척도법입니다. 데이터 군 내에서 특정 데이터가 가지는 위치를 볼 때 사용합니다.

$$ x_{new} = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$

### MinMaxScaler


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(iris.data)
iris_scaled = scaler.transform(iris.data)
iris_scaled[:5]
```




    array([[0.22222222, 0.625     , 0.06779661, 0.04166667],
           [0.16666667, 0.41666667, 0.06779661, 0.04166667],
           [0.11111111, 0.5       , 0.05084746, 0.04166667],
           [0.08333333, 0.45833333, 0.08474576, 0.04166667],
           [0.19444444, 0.66666667, 0.06779661, 0.04166667]])
