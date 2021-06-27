---
title:  "3. 기계학습(Machine Learning) 알고리즘"
toc: true
toc_sticky: true

categories:
  - time_Series
---


## 정규화 방법론(Regularized Method, Penalized Method, Contrained Least Squares)

> **"선형회귀 계수(Weight)에 대한 제약 조건을 추가함으로써 모형이 과도하게 최적화되는 현상, 즉 과최적화를 막는 방법"**  
> **"과최적화는 계수 크기를 과도하게 증가하는 경향이 있기에, 정규화 방법에서의 제약 조건은 일반적으로 계수의 크기를 제한하는 방법"**  

## 정규화 회귀분석 알고리즘

**0) Standard Regression:**


$\hat{\beta} = arg\underset{\hat{\beta}}{min} \Biggl[\displaystyle \sum_{j=1}^t \Bigl(y_j - \displaystyle \sum_{i=0}^k \beta_i x_{ij}\Bigr)^2\Biggr]$



**1) Ridge Regression:**  
- **정규화조건/패널티/제약조건:** 추정계수의 제곱합을 최소로 하는 것
$
\begin{aligned}
\hat{\beta} = arg\underset{\hat{\beta}}{min} \Biggl[\displaystyle \sum_{j=1}^t \Bigl(y_j - \displaystyle \sum_{i=0}^k \beta_i x_{ij}\Bigr)^2 + \lambda \displaystyle \sum_{i=0}^k \beta_i^2\Biggr] \\ where~\lambda~is~hyper~parameter(given~by~human)
\end{aligned}
$

- **하이퍼파라미터($\lambda$):** 기존의 잔차 제곱합과 추가 제약 조건의 비중을 조절하기 위한 하이퍼모수(hyperparameter)  
    - $\lambda$=0: 일반적인 선형 회귀모형(OLS)  
    - $\lambda$를 크게 두면 정규화(패널티) 정도가 커지기 때문에 가중치($\beta_i$)의 값들이 커질 수 없음(작아짐)  
    - $\lambda$를 작게 두면 정규화(패널티) 정도가 작아 지기 때문에 가중치($\beta_i$)의 값들의 자유도가 높아져 커질 수 있음(커짐)


**2) Lasso(Least Absolute Shrinkage and Selection Operator) Regression:**  
- **정규화조건/패널티/제약조건:** 추정계수의 절대값 합을 최소로 하는 것
$
\begin{aligned}
\hat{\beta} = arg\underset{\hat{\beta}}{min} \Biggl[\displaystyle \sum_{j=1}^t \Bigl(y_j - \displaystyle \sum_{i=0}^k \beta_i x_{ij}\Bigr)^2 + \lambda \displaystyle \sum_{i=0}^k \left|\beta_i \right|\Biggr] \\ where~\lambda~is~hyper~parameter(given~by~human)
\end{aligned}
$
![Ridge_Lasso](/assets/Ridge_Lasso.png)


**3) Elastic Net:**  
- **정규화조건/패널티/제약조건:** 추정계수의 절대값 합과 제곱합을 동시에 최소로 하는 것
$
\begin{aligned}
\hat{\beta} &= arg\underset{\hat{\beta}}{min} \Biggl[\displaystyle \sum_{j=1}^t \Bigl(y_j - \displaystyle \sum_{i=0}^k \beta_i x_{ij}\Bigr)^2 + \lambda_1 \displaystyle \sum_{i=0}^k \left|\beta_i \right| + \lambda_2 \displaystyle \sum_{i=0}^k \beta_i^2\Biggr] \\ &where~\lambda_1~and~\lambda_2~are~hyper~parameters(given~by~human)
\end{aligned}
$

### 하이퍼파라미터 특성 및 요약

- **최적 정규화(최적 하이퍼파라미터 추정):** 하이퍼파라미터(Hyperparameter)에 따른 검증성능 차이 존재
    - **Train Set:** 하이퍼파라미터가 작으면 작을수록 좋아짐(과최적화)
    - **Test Set:** 하이퍼파라미터가 특정한 범위에 있을때 좋아짐(추정필요)


- **Summary**  

> - **Standard:**
    ![Regression_Result_Standard](/assets/Regression_Result_Standard_qfpcbcp8r.png)
> - **Ridge:**
    - 알고리즘이 모든 변수들을 포함하려 하기 때문에 계수의 크기가 작아지고 모형의 복잡도가 줄어듬  
    - 모든 변수들을 포함하려 하므로 변수의 수가 많은 경우 효과가 좋지 않으나 과적합(Overfitting)을 방지하는데 효과적  
    - 다중공선성이 존재할 경우, 변수 간 상관관계에 따라 계수로 다중공선성이 분산되기에 효과가 높음  
    ![Regression_Result_Ridge1](/assets/Regression_Result_Ridge1.png)![Regression_Result_Ridge2](/assets/Regression_Result_Ridge2.png)   
> - **LASSO:**  
    - 알고리즘이 최소한의 변수를 포함하여 하기 때문의 나머지 변수들의 계수는 0이됨 (Feature Selection 기능)  
    - 변수선택 기능이 있기에 일반적으로 많이 사용되는 이점이 있지만 특정변수에 대한 계수가 커지는 단점 존재  
    - 다중공선성이 존재할 경우, 특정 변수만을 선택하는 방식이라 **Ridge**에 비해 다중공선성 문제에 효과가 낮음  
    ![Regression_Result_Lasso1](/assets/Regression_Result_Lasso1_s2k26xgjp.png)![Regression_Result_Lasso2](/assets/Regression_Result_Lasso2.png)
> - **Elastic Net:**  
    - 큰 데이터셋에서 Ridge와 LASSO의 효과를 모두 반영하기에 효과가 좋음 (적은 데이터셋은 효과 낮음)  
    ![Regression_Result_EN](/assets/Regression_Result_EN.png)

- **파라미터 세팅(실습)**
    > **1) "statsmodels":** 선형 회귀모형 클래스의 fit_regularized 메서드를 사용하여 Ridge/LASSO/Elastic Net 계수 추정
    - **Ridge:**
    <center>
    $\lambda_1 = 0,~~0 < \lambda_2 < 1 \\ => L_1 = 0,~~alpha \ne 0$
    </center>
    - **LASSO:**
    <center>
    $0 < \lambda_1 < 1,~~\lambda_2 = 0 \\ => L_1 = 1,~~alpha \ne 0$
    </center>
    - **Elastic Net:**
    <center>
    $0 < (\lambda_1, \lambda_2) < 1 \\ => 0 < L_1 < 1,~~alpha \ne 0$
    </center>

    > **2) "sklearn":** 정규화 회귀모형을 위한 Ridge, Lasso, ElasticNet 별도 클래스 제공
    - [**Ridge:**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    <center>
    $0 < (\lambda = alpha) < 1$
    </center>
    - [**LASSO:**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
    <center>
    $0 < (\lambda = alpha) < 1$
    </center>
    - [**Elastic Net:**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
    <center>
    $0 < (\lambda_1, \lambda_2) < 1 \\ => 0 < L_1 < 1,~~alpha \ne 0$
    </center>

~~~
# Ridge
fit = Ridge(alpha=0.5, fit_intercept=True, normalize=True, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)

# LASSO
fit = Lasso(alpha=0.5, fit_intercept=True, normalize=True, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)

# Elastic Net
fit = ElasticNet(alpha=0.01, l1_ratio=1, fit_intercept=True, normalize=True, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)
~~~

### 실습


```python
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
```


```python
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print('Data View')
display(pd.concat([pd.DataFrame(y, columns=['diabetes_value']), pd.DataFrame(X, columns=diabetes.feature_names)], axis=1).head())
```

    Data View



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_value</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151.0</td>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75.0</td>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>141.0</td>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206.0</td>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.0</td>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
  </tbody>
</table>
</div>



```python
alpha_weight = 0.5
fit = Ridge(alpha=alpha_weight, fit_intercept=True, normalize=True, random_state=123).fit(X, y)
pd.DataFrame(np.hstack([fit.intercept_, fit.coef_]), columns=['alpha = {}'.format(alpha_weight)])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha = 0.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>152.133484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.137357</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-131.242606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>383.481783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>244.837872</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-15.187056</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-58.344798</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-174.842798</td>
    </tr>
    <tr>
      <th>8</th>
      <td>121.985055</td>
    </tr>
    <tr>
      <th>9</th>
      <td>328.499702</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110.886036</td>
    </tr>
  </tbody>
</table>
</div>




```python
alpha_weight = 0.5
fit = Lasso(alpha=alpha_weight, fit_intercept=True, normalize=True, random_state=123).fit(X, y)
pd.DataFrame(np.hstack([fit.intercept_, fit.coef_]), columns=['alpha = {}'.format(alpha_weight)])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha = 0.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>152.133484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>471.038733</td>
    </tr>
    <tr>
      <th>4</th>
      <td>136.507108</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-58.319549</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>408.023324</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
result_Ridge = pd.DataFrame()
alpha_candidate = np.hstack([0, np.logspace(-2, 1, 4)])
for alpha_weight in alpha_candidate:
    fit = Ridge(alpha=alpha_weight, fit_intercept=True, normalize=True, random_state=123).fit(X, y)
    result_coef = pd.DataFrame(np.hstack([fit.intercept_, fit.coef_]), columns=['alpha = {}'.format(alpha_weight)])
    result_Ridge = pd.concat([result_Ridge, result_coef], axis=1)

result_LASSO = pd.DataFrame()
alpha_candidate = np.hstack([0, np.logspace(-2, 1, 4)])
for alpha_weight in alpha_candidate:
    fit = Lasso(alpha=alpha_weight, fit_intercept=True, normalize=True, random_state=123).fit(X, y)
    result_coef = pd.DataFrame(np.hstack([fit.intercept_, fit.coef_]), columns=['alpha = {}'.format(alpha_weight)])
    result_LASSO = pd.concat([result_LASSO, result_coef], axis=1)

result_Ridge.plot(figsize=(10,10), legend=True, ax=plt.subplot(211))
plt.title('Ridge')
plt.xlabel('Columns')
plt.ylabel('coefficients')
plt.legend(fontsize=13)
plt.grid()
result_LASSO.plot(legend=True, ax=plt.subplot(212))
plt.title('LASSO')
plt.xlabel('Columns')
plt.ylabel('coefficients')
plt.legend(fontsize=13)
plt.tight_layout()
plt.grid()
plt.show()
```



![output_9_0](/assets/output_9_0_ynnxfbll3.png)




```python
result_Ridge.T.plot(figsize=(10,10), legend=False, ax=plt.subplot(211))
plt.title('Ridge')
plt.xticks(np.arange(len(result_Ridge.columns)), [i for i in result_Ridge.columns])
plt.ylabel('coefficients')
plt.grid()
result_LASSO.T.plot(legend=False, ax=plt.subplot(212))
plt.title('LASSO')
plt.xticks(np.arange(len(result_Ridge.columns)), [i for i in result_Ridge.columns])
plt.ylabel('coefficients')
plt.tight_layout()
plt.grid()
plt.show()
```



![output_10_0](/assets/output_10_0.png)



## Bagging and Boosting 모델

### 편향-분산 상충관계(Bias-variance Trade-off)  

**1) 편향과 분산의 정의**
> **(비수학적 이해)**
- **편향(Bias):** 점추정  
    - 예측값과 실제값의 차이  
    - 모델 학습시 여러 데이터로 학습 후 예측값의 범위가 정답과 얼마나 멀리 있는지 측정  
- **편향(Bias(Real)):** 모형화(단순화)로 미처 반영하지 못한 복잡성  
    <U>=> 편향이 작다면 Training 데이터 패턴(복잡성)을 최대반영 의미</U>  
    <U>=> 편향이 크다면 Training 데이터 패턴(복잡성)을 최소반영 의미</U>  
- **분산(Variance):** 구간추정  
    - 학습한 모델의 예측값이 평균으로부터 퍼진 정도(변동성/분산)  
    - 여러 모델로 학습을 반복한다면, 학습된 모델별로 예측한 값들의 차이를 측정
- **분산(Variance(Real)):** 다른 데이터(Testing)를 사용했을때 발생할 변화  
    <U>=> 분산이 작다면 다른 데이터로 예측시 적은 변동 예상</U>  
    <U>=> 분산이 크다면 다른 데이터로 예측시 많은 변동 예상</U>  

![Bias_Variance1](/assets/Bias_Variance1.jpeg)

> **(수학적 이해)**

\begin{align*}
\text{Equation of Error} && Err(x) &= E\Bigl[\bigl(Y-\hat{f}(x)\bigr)^2 \Bigr] \\
&& &= \Bigl(E[\hat{f}(x)] - f(x)\Bigr)^2 + E \Bigl[\bigl(\hat{f}(x) - E[\hat{f}(x)]\bigr)^2 \Bigr] + \sigma_{\epsilon}^2 \\
&& &= \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\end{align*}

**2) 편향과 분산의 관계**
- **모델의 복잡도가 낮으면 Bias가 증가하고 Variance가 감소(Underfitting)**  
: 구간추정 범위는 좁으나 점추정 정확성 낮음  
: Training/Testing 모두 예측력이 낮음
- **모델의 복잡도가 높으면 Bias가 감소하고 Variance가 증가(Overfitting)**  
: 점추정 정확성은 높으나 구간추정 범위는 넓음  
: Training만 잘 예측력 높고 Testing은 예측력 낮음  
- **Bias와 Variance가 최소화 되는 수준에서 모델의 복잡도 선택**  

![Bias-Variance-Tradeoff](/assets/Bias-Variance-Tradeoff.png)
![Bias_Variance4](/assets/Bias_Variance4.png)

**3) 편향과 분산 모두를 최소화하는 방법**
<center><img src='Image/Bias_Variance_Reduce.png' width='600'></center>

### Bagging vs Boosting

> **앙상블(Ensemble, Ensemble Learning, Ensemble Method)이란 머신러닝에서 여러개의 모델을 학습시켜,  
그 모델들의 예측결과들을 이용해 하나의 모델보다 더 나은 값을 예측하는 방법**

- **Bagging(Bootstrap Aggregating):**   
    - 부트스트래핑(Bootstraping): 예측값과 실제값의 차이 중복을 허용한 리샘플링(Resampling)  
    - 페이스팅(Pasting): 이와 반대로 중복을 허용하지 않는 샘플링  


- **Boosting:**   
    - 성능이 약한 학습기(weak learner)를 여러 개 연결하여 강한 학습기(strong learner)를 만드는 앙상블 학습  
    - 앞에서 학습된 모델을 보완해나가면서 더나은 모델로 학습시키는 것  

<center><img src='Image/Bagging_Boosting.png' width='700'></center>

| - | Bagging | Boosting |
|-------------|---------------------------------------|-----------------------------------------|
| 특징 | 병렬 앙상블 모델(각 모델은 서로 독립) | 연속 앙상블 모델(이전 모델의 오류 반영) |
| 목적 | Variance 감소 | Bias 감소 |
| 적합한 상황 | Low Bias + High Variance | High Bias + Low Variance |
| Sampling | Random Sampling | Random Sampling with weight on error |

### Bagging 알고리즘

- **의사결정나무(Decision Tree):**  

![Bagging_DT](/assets/Bagging_DT.png)

- **렌덤포레스트(Random Forest):** 여러개의 의사결정나무(Decision Tree) 생성한 다음, 각 개별 트리의 예측값들 중 가장 많은 선택을 받은 변수들로 예측하는 알고리즘, 의사결정나무의 CLT버전

![Bagging_RF](/assets/Bagging_RF.jpg)

~~~
# DecisionTree
fit = DecisionTreeRegressor().fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)

# RandomForestRegressor
fit = RandomForestRegressor(n_estimators=100, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)
~~~

### Boosting 알고리즘

- **Adaptive Boosting(AdaBoost):** 학습된 모델이 과소적합(학습하기 어려운 데이터)된 학습 데이터 샘플의 가중치를 높이면서 더 잘 적합되도록 하는 방식

![Boosting_AdaBoost](/assets/Boosting_AdaBoost.png)

- **Gradient Boosting Machine(GBM):** 아다부스트 처럼 학습단계 마다 데이터 샘플의 가중치를 업데이트 하는 것이 아니라, 학습 전단계 모델에서의 잔차(Residual)을 모델에 학습시키는 방법

![Boosting_GBM](/assets/Boosting_GBM.png)

- **XGBoost(eXtreme Gradient Boosting):** 높은 예측력으로 많은 양의 데이터를 다룰 때 사용되는 부스팅 알고리즘  

![Boosting_XGBoost](/assets/Boosting_XGBoost.png)

- **LightGBM:** 현존하는 부스팅 알고리즘 중 가장 빠르고 높은 예측력 제공

![Boosting_LightGBM](/assets/Boosting_LightGBM.png)

| Algorithms | Specification | Others |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| AdaBoost | 다수결을 통한 정답분류 및 오답에 가중치 부여 | - |
| GBM | 손실함수(검증지표)의 Gradient로 오답에 가중치 부여 | - |
| XGBoost | GMB대비 성능향상<br/>시스템(CPU, Mem.) 자원 효율적 사용 | 2014년 공개 |
| LightGBM | XGBoost대비 성능향상 및 자원소모 최소화<br/>XGBoost가 처리하지 못하는 대용량 데이터 학습가능<br/>근사치분할(Approximates the Split)을 통한 성능향상 | 2016년 공개 |

~~~
# GradientBoostingRegression
fit = GradientBoostingRegressor(alpha=0.1, learning_rate=0.05, loss='huber', criterion='friedman_mse',
                                           n_estimators=1000, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)

# XGBoost
fit = XGBRegressor(learning_rate=0.05, n_estimators=100, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)

# LightGMB
fit = LGBMRegressor(learning_rate=0.05, n_estimators=100, random_state=123).fit(X_train, Y_train)
pred_tr = fit.predict(X_train)
pred_te = fit.predict(X_test)
~~~

### 비교

![Bagging_Boosting2](/assets/Bagging_Boosting2.png)

## 회귀분석 알고리즘 정리

- **변수 세팅에 따른 분류:**

![Regression-Algorithms-Tree1](/assets/Regression-Algorithms-Tree1.png)

- **문제 해결에 따른 분류:**

![Regression-Algorithms-Tree2](/assets/Regression-Algorithms-Tree2.png)
