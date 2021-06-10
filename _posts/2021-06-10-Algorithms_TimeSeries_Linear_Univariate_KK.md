---
title:  "4. 시계열 알고리즘"
toc: true
toc_sticky: true

categories:
  - time_Series
---

![TS_Algorithm_Concept](/assets/TS_Algorithm_Concept.png)

## 비정상성(Non-stationary)의 정상성(Stationary) 변환

> - **목적: 정상성 확보를 통해 안정성이 높아지고 예측력 향상**
> - **장점: 절약성 원칙(Principle of Parsimony)에 따라 적은 모수만으로 모델링 가능하기에 과적합 확률이 줄어듬**
> - **방법: 제곱, 루트, 로그, 차분 등**


- **이론예시:**
![Stationary_Example](/assets/Stationary_Example.png)

    - Trend: a/c/e/f/i  
    - Seasonality: d/h/i  
    - Cycle: g
    - Non-constant Variance: i  

**1) 로그변환(Logarithm Transform):**
> - 시간흐름에 비례하여 값이 커지는 경우(분산 증가)
> - 비정상 확률 과정으로 표준편차가 자료의 크기에 비례하여 증가하거나 지수함수적으로 증가하는 경우
> - 로그 변환한 확률 과정의 분산은 일정하기에 추세 제거로 기댓값이 0이 되면 정상 과정으로 모형화 가능

\begin{align*}
\text{Distribution of Original} && \text{E}(Y_t) &= \mu_t = f(t) \\
&& \sqrt{\text{Var}(Y_t)} &= \mu_t \sigma \\
\text{Distribution of Log-transform} && Y_t &= Y_{t-1} + Y_t - Y_{t-1} \\
&& \dfrac{Y_t}{Y_{t-1}} &= 1 + \dfrac{Y_t - Y_{t-1}}{Y_{t-1}} \\
&& log(\dfrac{Y_t}{Y_{t-1}}) &= log(1 + \dfrac{Y_t - Y_{t-1}}{Y_{t-1}}) \approx \dfrac{Y_t - Y_{t-1}}{Y_{t-1}} \\
&& log(Y_t) - log(Y_{t-1}) &\approx \dfrac{Y_t - Y_{t-1}}{Y_{t-1}} \\
&& log(Y_t) &\approx log(Y_{t-1}) + \dfrac{Y_t - Y_{t-1}}{Y_{t-1}} \\
&& log(Y_t) &\approx log(\mu_t) + \dfrac{Y_t - \mu_t}{\mu_t} \\
&& \text{E}(\log Y_t) &= \log \mu_t \\
&& \text{Var}(\log Y_t) &\approx \sigma^2 \\
\text{*Generalization of Return} && R_t &= \dfrac{Y_{t}}{Y_{t-1}} - 1 \\
&& \log{Y_t} - \log{Y_{t-1}} &= \log{R_t + 1} \approx R_t \;\; \text{ if } \left| R_t \right| < 0.2 \\
\end{align*}

**2) 차분(Difference):** 특정 시점 또는 시점들의 데이터가 발산할 경우 시점간 차분(변화량)으로 정상성 변환 가능
- **계절성(Seasonality, $S_t$)**: 특정한 달/요일에 따라 기대값이 달라지는 것, 변수 더미화를 통해 추정 가능
> - **계절성 제거: 1) 계절성 추정($f(t)$) 후 계절성 제거를 통한 정상성 확보 (수학적 이해)**
    - 확률과정의 계절변수 더미화를 통해 기댓값 함수를 알아내는 것
    - 확률과정($Y_t$)이 추정이 가능한 결정론적 계절성함수($f(t)$)와 정상확률과정($Y^s_t$)의 합

    \begin{align*}
    \text{Main Equation} && Y_t &= f(t) + Y^s_t \\
    \text{where} && f(t) &= \sum_{i=0}^{\infty} a_i D_i = a_0 + a_1 D_1 + a_2 D_2 + \cdots
    \end{align*}
> - **계절성 제거: 2) 차분 적용 $(1-L^d) Y_t$ 후 계절성 제거를 통한 정상성 확보 (수학적 이해)**

    \begin{align*}
    \text{Main Equation of d=1} && Y_t &=> (1-L^1) Y_t \\
    && &= (1-Lag^1) Y_t \\
    && &= Y_t - Lag^1(Y_t) \\
    && &= Y_t - Y_{t-1} \\
    \text{Main Equation of d=2} && Y_t &=> (1-L^2) Y_t \\
    && &= (1-Lag^2) Y_t \\
    && &= Y_t - Lag^2(Y_t) \\
    && &= Y_t - Y_{t-2} \\
    \end{align*}


- **추세(Trend, $T_t$)**: 시계열이 시간에 따라 증가, 감소 또는 일정 수준을 유지하는 경우  
> - **추세 제거: 1) 추세 추정($f(t)$) 후 추세 제거를 통한 정상성 확보 (수학적 이해)**
    - 확률과정의 결정론적 기댓값 함수를 알아내는 것
    - 확률과정($Y_t$)이 추정이 가능한 결정론적 추세함수($f(t)$)와 정상확률과정($Y^s_t$)의 합

    \begin{align*}
    \text{Main Equation} && Y_t &= f(t) + Y^s_t \\
    \text{where} && f(t) &= \sum_{i=0}^{\infty} a_i t^i = a_0 + a_1 t + a_2 t^2 + \cdots
    \end{align*}
> - **추세 제거: 2) 차분 적용 $(1-L^1)^d Y_t$ 후 추세 제거를 통한 정상성 확보 (수학적 이해)**

    \begin{align*}
    \text{Main Equation of d=1} && Y_t &=> (1-L^1)^1 Y_t \\
    && &= (1-Lag^1)^1 Y_t \\
    && &= Y_t - Lag^1(Y_t) \\
    && &= Y_t - Y_{t-1} \\
    \text{Main Equation of d=2} && Y_t &=> (1-L^1)^2 Y_t \\
    && &= (1-2L^1+L^2) Y_t \\
    && &= (1-2Lag^1+Lag^2) Y_t \\
    && &= Y_t - 2Lag^1(Y_t) + Lag^2(Y_t) \\
    && &= Y_t - Lag^1(Y_t) - Lag^1(Y_t) + Lag^2(Y_t) \\
    && &= (Y_t - Lag^1(Y_t)) - (Lag^1(Y_t) - Lag^2(Y_t)) \\
    && &= (Y_t - L^1(Y_t)) - (L^1(Y_t) - L^2(Y_t)) \\
    && &= (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2}) \\
    && &= Y_t - 2Y_{t-1} + Y_{t-2} \\
    \end{align*}

**3) Box-Cox 변환:** 정규분포가 아닌 자료를 정규분포로 변환하기 위해 사용
> - 모수(parameter) $\lambda$를 가지며, 보통 여러가지 $\lambda$ 값을 시도하여 가장 정규성을 높여주는 값을 사용

\begin{align*}
y^{(\boldsymbol{\lambda})} =
\begin{cases}
\dfrac{y^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0, \\
\ln{y} & \text{if } \lambda = 0,
\end{cases}
\end{align*}

### 정상성 테스트 방향

> **추세와 계절성 모두 제거된 정상성 데이터 변환 필요!**
>> - **ADF 정상성 확인 -> 추세 제거 확인 Measure**  
: ADF 검정통계량은 정상이라고 해도 데이터에 계절성이 포함되면 ACF의 비정상 Lag 존재하는 비정상데이터 가능  

>> - **KPSS 정상성 확인 -> 계절성 제거 확인 Measure**  
: KPSS 검정통계량은 정상이라고 해도 데이터에 추세가 포함되면 ACF의 비정상 Lag 존재하는 비정상데이터 가능  


### 실습: 대기중 CO2농도 추세 제거


```python
# 라이브러리 및 데이터 로딩
import pandas as pd
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

raw_set = datasets.get_rdataset("CO2", package="datasets")
raw = raw_set.data
```


```python
# 데이터 확인 및 추세 추정 (선형)
display(raw.head())
plt.plot(raw.time, raw.value)
plt.show()

result = sm.OLS.from_formula(formula='value~time', data=raw).fit()
display(result.summary())

trend = result.params[0] + result.params[1] * raw.time
plt.plot(raw.time, raw.value, raw.time, trend)
plt.show()
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
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1,959.00</td>
      <td>315.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1,959.08</td>
      <td>316.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1,959.17</td>
      <td>316.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1,959.25</td>
      <td>317.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1,959.33</td>
      <td>318.13</td>
    </tr>
  </tbody>
</table>
</div>




![output_5_1](/assets/output_5_1.png)




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>value</td>      <th>  R-squared:         </th> <td>   0.969</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.969</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.479e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 27 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:29:09</td>     <th>  Log-Likelihood:    </th> <td> -1113.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   468</td>      <th>  AIC:               </th> <td>   2231.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   466</td>      <th>  BIC:               </th> <td>   2239.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>-2249.7742</td> <td>   21.268</td> <td> -105.784</td> <td> 0.000</td> <td>-2291.566</td> <td>-2207.982</td>
</tr>
<tr>
  <th>time</th>      <td>    1.3075</td> <td>    0.011</td> <td>  121.634</td> <td> 0.000</td> <td>    1.286</td> <td>    1.329</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15.857</td> <th>  Durbin-Watson:     </th> <td>   0.212</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>   7.798</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.048</td> <th>  Prob(JB):          </th> <td>  0.0203</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.375</td> <th>  Cond. No.          </th> <td>3.48e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.48e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




![output_5_3](/assets/output_5_3.png)




```python
# 데이터 확인 및 추세 추정 (비선형)
display(raw.head())
plt.plot(raw.time, raw.value)
plt.show()

result = sm.OLS.from_formula(formula='value~time+I(time**2)', data=raw).fit()
display(result.summary())

trend = result.params[0] + result.params[1] * raw.time + result.params[2] * raw.time**2
plt.plot(raw.time, raw.value, raw.time, trend)
plt.show()
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
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1,959.00</td>
      <td>315.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1,959.08</td>
      <td>316.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1,959.17</td>
      <td>316.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1,959.25</td>
      <td>317.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1,959.33</td>
      <td>318.13</td>
    </tr>
  </tbody>
</table>
</div>




![output_6_1](/assets/output_6_1.png)




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>value</td>      <th>  R-squared:         </th> <td>   0.979</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.979</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.075e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 27 Sep 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:30:54</td>     <th>  Log-Likelihood:    </th> <td> -1027.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   468</td>      <th>  AIC:               </th> <td>   2062.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   465</td>      <th>  BIC:               </th> <td>   2074.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  4.77e+04</td> <td> 3482.902</td> <td>   13.696</td> <td> 0.000</td> <td> 4.09e+04</td> <td> 5.45e+04</td>
</tr>
<tr>
  <th>time</th>         <td>  -49.1907</td> <td>    3.521</td> <td>  -13.971</td> <td> 0.000</td> <td>  -56.110</td> <td>  -42.272</td>
</tr>
<tr>
  <th>I(time ** 2)</th> <td>    0.0128</td> <td>    0.001</td> <td>   14.342</td> <td> 0.000</td> <td>    0.011</td> <td>    0.015</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>66.659</td> <th>  Durbin-Watson:     </th> <td>   0.306</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  17.850</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.116</td> <th>  Prob(JB):          </th> <td>0.000133</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.072</td> <th>  Cond. No.          </th> <td>1.35e+11</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.35e+11. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




![output_6_3](/assets/output_6_3.png)




```python
# 추세 제거 및 정상성 확인
## 방법1
plt.plot(raw.time, result.resid)
plt.show()

display(stationarity_adf_test(result.resid, []))
display(stationarity_kpss_test(result.resid, []))

sm.graphics.tsa.plot_acf(result.resid, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()
```



![output_7_0](/assets/output_7_0.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-2.53</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.11</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>454.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.44</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>260.10</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.17</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>18.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_7_3](/assets/output_7_3.png)




```python
raw.value.diff(1)
```




    0       nan
    1      0.89
    2      0.19
    3      1.06
    4      0.57
           ...
    463   -1.95
    464   -2.33
    465    0.59
    466    1.66
    467    1.85
    Name: value, Length: 468, dtype: float64




```python
# 추세 제거 및 정상성 확인
## 방법2
plt.plot(raw.time[1:], raw.value.diff(1).dropna())
plt.show()

display(stationarity_adf_test(raw.value.diff(1).dropna(), []))
display(stationarity_kpss_test(raw.value.diff(1).dropna(), []))

sm.graphics.tsa.plot_acf(raw.value.diff(1).dropna(), lags=100, use_vlines=True)
plt.tight_layout()
plt.show()
```



![output_9_0](/assets/output_9_0_6qgd6h8iu.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-5.14</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>454.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.44</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>271.87</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.04</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>18.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_9_3](/assets/output_9_3.png)



### 실습: 호흡기질환 사망자수 계절성 제거


```python
# 라이브러리 및 데이터 로딩
import pandas as pd
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

raw_set = datasets.get_rdataset("deaths", package="MASS")
raw = raw_set.data
```


```python
# 시간변수 추출
raw.time = pd.date_range('1974-01-01', periods=len(raw), freq='M')
raw['month'] = raw.time.dt.month
```


```python
# 데이터 확인 및 추세 추정
display(raw.tail())
plt.plot(raw.time, raw.value)
plt.show()

display(stationarity_adf_test(raw.value, []))
display(stationarity_kpss_test(raw.value, []))
sm.graphics.tsa.plot_acf(raw.value, lags=50, use_vlines=True, title='ACF')
plt.tight_layout()
plt.show()

result = sm.OLS.from_formula(formula='value ~ C(month) - 1', data=raw).fit()
display(result.summary())

plt.plot(raw.time, raw.value, raw.time, result.fittedvalues)
plt.show()
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
      <th>time</th>
      <th>value</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>1979-08-31</td>
      <td>1354</td>
      <td>8</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1979-09-30</td>
      <td>1333</td>
      <td>9</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1979-10-31</td>
      <td>1492</td>
      <td>10</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1979-11-30</td>
      <td>1781</td>
      <td>11</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1979-12-31</td>
      <td>1915</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




![output_13_1](/assets/output_13_1.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-0.57</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.88</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>59.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.55</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>841.38</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.65</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_13_4](/assets/output_13_4.png)




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>value</td>      <th>  R-squared:         </th> <td>   0.853</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.826</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   31.66</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 27 Sep 2020</td> <th>  Prob (F-statistic):</th> <td>6.55e-21</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:50:37</td>     <th>  Log-Likelihood:    </th> <td> -494.38</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    72</td>      <th>  AIC:               </th> <td>   1013.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    60</td>      <th>  BIC:               </th> <td>   1040.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>C(month)[1]</th>  <td> 2959.3333</td> <td>  103.831</td> <td>   28.502</td> <td> 0.000</td> <td> 2751.641</td> <td> 3167.025</td>
</tr>
<tr>
  <th>C(month)[2]</th>  <td> 2894.6667</td> <td>  103.831</td> <td>   27.879</td> <td> 0.000</td> <td> 2686.975</td> <td> 3102.359</td>
</tr>
<tr>
  <th>C(month)[3]</th>  <td> 2743.0000</td> <td>  103.831</td> <td>   26.418</td> <td> 0.000</td> <td> 2535.308</td> <td> 2950.692</td>
</tr>
<tr>
  <th>C(month)[4]</th>  <td> 2269.6667</td> <td>  103.831</td> <td>   21.859</td> <td> 0.000</td> <td> 2061.975</td> <td> 2477.359</td>
</tr>
<tr>
  <th>C(month)[5]</th>  <td> 1805.1667</td> <td>  103.831</td> <td>   17.386</td> <td> 0.000</td> <td> 1597.475</td> <td> 2012.859</td>
</tr>
<tr>
  <th>C(month)[6]</th>  <td> 1608.6667</td> <td>  103.831</td> <td>   15.493</td> <td> 0.000</td> <td> 1400.975</td> <td> 1816.359</td>
</tr>
<tr>
  <th>C(month)[7]</th>  <td> 1550.8333</td> <td>  103.831</td> <td>   14.936</td> <td> 0.000</td> <td> 1343.141</td> <td> 1758.525</td>
</tr>
<tr>
  <th>C(month)[8]</th>  <td> 1408.3333</td> <td>  103.831</td> <td>   13.564</td> <td> 0.000</td> <td> 1200.641</td> <td> 1616.025</td>
</tr>
<tr>
  <th>C(month)[9]</th>  <td> 1397.3333</td> <td>  103.831</td> <td>   13.458</td> <td> 0.000</td> <td> 1189.641</td> <td> 1605.025</td>
</tr>
<tr>
  <th>C(month)[10]</th> <td> 1690.0000</td> <td>  103.831</td> <td>   16.277</td> <td> 0.000</td> <td> 1482.308</td> <td> 1897.692</td>
</tr>
<tr>
  <th>C(month)[11]</th> <td> 1874.0000</td> <td>  103.831</td> <td>   18.049</td> <td> 0.000</td> <td> 1666.308</td> <td> 2081.692</td>
</tr>
<tr>
  <th>C(month)[12]</th> <td> 2478.5000</td> <td>  103.831</td> <td>   23.871</td> <td> 0.000</td> <td> 2270.808</td> <td> 2686.192</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>19.630</td> <th>  Durbin-Watson:     </th> <td>   1.374</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  49.630</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.787</td> <th>  Prob(JB):          </th> <td>1.67e-11</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.750</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




![output_13_6](/assets/output_13_6.png)




```python
# 추세 제거 및 정상성 확인
## 방법1
plt.plot(raw.time, result.resid)
plt.show()

display(stationarity_adf_test(result.resid, []))
display(stationarity_kpss_test(result.resid, []))
sm.graphics.tsa.plot_acf(result.resid, lags=50, use_vlines=True)
plt.tight_layout()
plt.show()
```



![output_14_0](/assets/output_14_0.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-5.84</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>71.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.53</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>812.36</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.54</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_14_3](/assets/output_14_3.png)




```python
# 계절성 제거 및 정상성 확인
## 방법2
sm.graphics.tsa.plot_acf(raw.value, lags=50, use_vlines=True)
plt.show()

plt.plot(raw.time, raw.value)
plt.title('Raw')
plt.show()
seasonal_lag = 3
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
seasonal_lag = 6
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
seasonal_lag = 12
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
plt.title('Lagged')
plt.legend()
plt.show()

seasonal_lag = 6
display(stationarity_adf_test(raw.value.diff(seasonal_lag).dropna(), []))
display(stationarity_kpss_test(raw.value.diff(seasonal_lag).dropna(), []))
sm.graphics.tsa.plot_acf(raw.value.diff(seasonal_lag).dropna(), lags=50,
                         use_vlines=True, title='ACF of Lag{}'.format(seasonal_lag))
plt.tight_layout()
plt.show()

seasonal_lag = 12
display(stationarity_adf_test(raw.value.diff(seasonal_lag).dropna(), []))
display(stationarity_kpss_test(raw.value.diff(seasonal_lag).dropna(), []))
sm.graphics.tsa.plot_acf(raw.value.diff(seasonal_lag).dropna(), lags=50,
                         use_vlines=True, title='ACF of Lag{}'.format(seasonal_lag))
plt.tight_layout()
plt.show()
```



![output_15_0](/assets/output_15_0.png)





![output_15_1](/assets/output_15_1.png)





![output_15_2](/assets/output_15_2.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-4.30</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>54.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.56</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>786.67</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.35</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_15_5](/assets/output_15_5.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-2.14</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.23</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>48.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.57</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>703.72</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.09</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_15_8](/assets/output_15_8.png)



### 실습: 랜덤워크의 정상성 변환


```python
# 라이브러리 호출
import pandas as pd
import numpy as np
import statsmodels.api as sm
from random import seed, random
import matplotlib.pyplot as plt
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

# 랜덤워크 데이터 생성
plt.figure(figsize=(10, 4))
seed(1)
random_walk = [-1 if random() < 0.5 else 1]
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
plt.plot(random_walk)
plt.tight_layout()
plt.show()

# 차분 전 랜덤워크 정상성 테스트
display('Before a difference:')
display(stationarity_adf_test(random_walk, []))
display(stationarity_kpss_test(random_walk, []))
sm.graphics.tsa.plot_acf(random_walk, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

# 차분 후 랜덤워크 정상성 테스트
display('After a difference:')
display(stationarity_adf_test(pd.Series(random_walk).diff(1).dropna(), []))
display(stationarity_kpss_test(pd.Series(random_walk).diff(1).dropna(), []))
sm.graphics.tsa.plot_acf(pd.Series(random_walk).diff(1).dropna(), lags=100, use_vlines=True)
plt.tight_layout()
plt.show()
```



![output_17_0](/assets/output_17_0.png)




    'Before a difference:'



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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.34</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.98</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>999.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.44</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>2,773.39</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>3.75</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>22.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_17_4](/assets/output_17_4.png)




    'After a difference:'



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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-31.08</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>998.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.44</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>2,770.18</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.22</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>22.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_17_8](/assets/output_17_8.png)




```python
# 랜덤워크 데이터 생성 및 통계량 Test
plt.figure(figsize=(10, 4))
seed(1)
rho = 0
random_walk = [-1 if random() < 0.5 else 1]
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = rho * random_walk[i-1] + movement
    random_walk.append(value)
plt.plot(random_walk)
plt.title('Rho: {}\n ADF p-value: {}'.format(rho, np.ravel(stationarity_adf_test(random_walk, []))[1]))
plt.tight_layout()
plt.show()

# rho 값을 변화시키면서 언제 비정상이 되는지 파악?
# 정상성의 계수 범위 추론 가능!
```



![output_18_0](/assets/output_18_0.png)



### 실습: 항공사 승객수요 스케일 변환(Log / Box-Cox)


```python
# 라이브러리 호출
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sp
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

# 데이터 준비
data = sm.datasets.get_rdataset("AirPassengers")
raw = data.data.copy()

# Box-Cox 변환 모수 추정
# 정규분포의 특정 범위(x)에서 lambda를 바꿔가며 정규성(measure:y)이 가장 높은 lambda(l_opt)를 선정
x, y = sp.stats.boxcox_normplot(raw.value, la=-3, lb=3)
y_transfer, l_opt = sp.stats.boxcox(raw.value)
print('Optimal Lambda: ', l_opt)

plt.plot(x, y)
plt.axvline(x=l_opt, color='r', ls="--")
plt.tight_layout()
plt.show()
```

    Optimal Lambda:  0.14802265137037945




![output_20_1](/assets/output_20_1.png)




```python
plt.figure(figsize=(12,4))
sm.qqplot(raw.value, fit=True, line='45', ax=plt.subplot(131))
plt.title('Y')
sm.qqplot(np.log(raw.value), fit=True, line='45', ax=plt.subplot(132))
plt.title('Log(Y)')
sm.qqplot(y_transfer, fit=True, line='45', ax=plt.subplot(133))
plt.title('BoxCox(Y)')
plt.tight_layout()
plt.show()
```



![output_21_0](/assets/output_21_0.png)



### 실습: 항공사 승객수요 정상성 변환


```python
# 라이브러리 호출
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

# 데이터 준비
data = sm.datasets.get_rdataset("AirPassengers")
raw = data.data.copy()

# 데이터 전처리
## 시간 인덱싱
if 'time' in raw.columns:
    raw.index = pd.date_range(start='1/1/1949', periods=len(raw['time']), freq='M')
    del raw['time']

## 정상성 확보
plt.figure(figsize=(12,8))
raw.plot(ax=plt.subplot(221), title='Y', legend=False)
np.log(raw).plot(ax=plt.subplot(222), title='log(Y)', legend=False)
raw.diff(1).plot(ax=plt.subplot(223), title='diff1(Y)', legend=False)
np.log(raw).diff(1).plot(ax=plt.subplot(224), title='diff1(log(Y))', legend=False)
plt.show()
```



![output_23_0](/assets/output_23_0.png)




```python
## 정상성 테스트
### 미변환
display('Non-transfer:')
plt.figure(figsize=(12,8))
raw.plot(ax=plt.subplot(222), title='Y', legend=False)
plt.show()

candidate_none = raw.copy()
display(stationarity_adf_test(candidate_none.values.flatten(), []))
display(stationarity_kpss_test(candidate_none.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_none, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

### 로그 변환
display('Log transfer:')
plt.figure(figsize=(12,8))
np.log(raw).plot(ax=plt.subplot(222), title='log(Y)', legend=False)
plt.show()

candidate_trend = np.log(raw).copy()
display(stationarity_adf_test(candidate_trend.values.flatten(), []))
display(stationarity_kpss_test(candidate_trend.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_trend, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

trend_diff_order_initial = 0
result = stationarity_adf_test(candidate_trend.values.flatten(), []).T
if result['p-value'].values.flatten() < 0.1:
    trend_diff_order = trend_diff_order_initial
else:
    trend_diff_order = trend_diff_order_initial + 1
print('Trend Difference: ', trend_diff_order)

### 로그+추세차분 변환
display('Log and trend diffrence transfer:')
plt.figure(figsize=(12,8))
np.log(raw).diff(trend_diff_order).plot(ax=plt.subplot(224), title='diff1(log(Y))', legend=False)
plt.show()

candidate_seasonal = candidate_trend.diff(trend_diff_order).dropna().copy()
display(stationarity_adf_test(candidate_seasonal.values.flatten(), []))
display(stationarity_kpss_test(candidate_seasonal.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_seasonal, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

seasonal_diff_order = sm.tsa.acf(candidate_seasonal)[1:].argmax() + 1
print('Seasonal Difference: ', seasonal_diff_order)

### 로그+추세차분+계절차분 변환
display('Log and trend+seasonal diffrence transfer:')
candidate_final = candidate_seasonal.diff(seasonal_diff_order).dropna().copy()
display(stationarity_adf_test(candidate_final.values.flatten(), []))
display(stationarity_kpss_test(candidate_final.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_final, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()
```


    'Non-transfer:'




![output_24_1](/assets/output_24_1.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.82</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.99</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>130.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>996.69</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>1.05</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_24_4](/assets/output_24_4.png)




    'Log transfer:'




![output_24_6](/assets/output_24_6.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-1.72</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.42</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>130.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-445.40</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>1.05</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_24_9](/assets/output_24_9.png)



    Trend Difference:  1



    'Log and trend diffrence transfer:'




![output_24_12](/assets/output_24_12.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-2.72</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>128.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-440.36</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_24_15](/assets/output_24_15.png)



    Seasonal Difference:  12



    'Log and trend+seasonal diffrence transfer:'



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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-4.44</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>118.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.49</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-415.56</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.11</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_24_20](/assets/output_24_20.png)



## 일반 선형확률과정(General Linear Process)

> **"시계열 데이터가 가우시안 백색잡음의 현재값과 과거값의 선형조합"**  

\begin{align*}
Y_t = \epsilon_t + \psi_1\epsilon_{t-1} + \psi_2\epsilon_{t-2} + \cdots \\
\end{align*}
\begin{align*}
where~\epsilon_i \sim i.i.d.~WN(0, \sigma_{\epsilon_i}^2)~and~\displaystyle \sum_{i=1}^{\infty}\psi_i^2 < \infty
\end{align*}

- **세부 알고리즘:**
    - WN(White Noise)
    - MA(Moving Average)
    - AR(Auto-Regressive)
    - ARMA(Auto-Regressive Moving Average)
    - ARIMA(Auto-Regressive Integrated Moving Average)
    - SARIMA(Seasonal ARIMA)

### WN(White Noise)

![White_Noise](/assets/White_Noise.png)

>**1) 잔차들은 정규분포이고, (unbiased) 평균 0과 일정한 분산을 가져야 함:**  
\begin{align*}
\{\epsilon_t : t = \dots, -2, -1, 0, 1, 2, \dots\} \sim N(0,\sigma^2_{\epsilon_t}) \\
\end{align*}
\begin{align*}
where~~ \epsilon_t \sim  i.i.d(independent~and~identically~distributed) \\
\end{align*}
\begin{align*}
\epsilon_t = Y_t - \hat{Y_t}, \;\; E(\epsilon_t) = 0, \;\; Var(\epsilon_t) = \sigma^2_{\epsilon_t} \\
\end{align*}
\begin{align*}
Cov(\epsilon_s, \epsilon_k) = 0~for~different~times!(s \ne k)
\end{align*}

>**2) 잔차들이 시간의 흐름에 따라 상관성이 없어야 함:**  
- 자기상관함수(Autocorrelation Fundtion([ACF](https://en.wikipedia.org/wiki/Autocorrelation)))를 통해 $Autocorrelation~=~0$인지 확인
    - 공분산(Covariance):
    <center>$Cov(Y_s, Y_k)$ = $E[(Y_s-E(Y_s))$$(Y_k-E(Y_k))]$ = $\gamma_{s,k}$</center>
    - 자기상관함수(Autocorrelation Function):
    <center>$Corr(Y_s, Y_k)$ = $\dfrac{Cov(Y_s, Y_k)}{\sqrt{Var(Y_s)Var(Y_k)}}$ = $\dfrac{\gamma_{s,k}}{\sqrt{\gamma_s \gamma_k}}$</center>
    - 편자기상관함수(Partial Autocorrelation Function): $s$와 $k$사이의 상관성을 제거한 자기상관함수
    <center>$Corr[(Y_s-\hat{Y}_s, Y_{s-t}-\hat{Y}_{s-t})]$ for $1<t<k$</center>

- **특성요약:**
    - 강정상 과정(Stictly Stationary Process)
    - 강정상 예시로 시계열분석 기본알고리즘 중 가장 중요함
    - 시차(lag)가 0일 경우, 자기공분산은 확률 분포의 분산이 되고 시차가 0이 아닌 경우, 자기공분산은 0.  
    \begin{align*}
    \gamma_i = \begin{cases} \text{Var}(\epsilon_t) & \;\; \text{ for } i = 0 \\  
    0 & \;\; \text{ for }  i \neq 0 \end{cases}
    \end{align*}
    - 시차(lag)가 0일 경우, 자기상관계수는 1이 되고 시차가 0이 아닌 경우, 자기상관계수는 0.
    \begin{align*}
    \rho_i = \begin{cases} 1 & \;\; \text{ for } i = 0 \\  
    0 & \;\; \text{ for }  i \neq 0 \end{cases}
    \end{align*}

**1) 예시: 가우시안 백색잡음**


```python
from scipy import stats
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(stats.norm.rvs(size=300))
plt.tight_layout()
plt.show()
```



![output_28_0](/assets/output_28_0.png)



**2) 예시: 베르누이 백색잡음**

> 백색잡음의 기반 확률분포가 반드시 정규분포일 필요는 없음


```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
samples = stats.bernoulli.rvs(0.5, size=100) * 2 - 1
plt.step(np.arange(len(samples)), samples)
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()
```



![output_30_0](/assets/output_30_0.png)



### MA(Moving Average)

> **"$MA(q)$: 알고리즘의 차수($q$)가 유한한 가우시안 백색잡음과정의 선형조합"**
- Exponential Smoothing 내  Moving Average Smoothing은 과거의 Trend-Cycle을 추정하기 위함이고, MA는 미래 값을 예측하기 위함

\begin{align*}
Y_t &= \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q}
\end{align*}
\begin{align*}
where~\epsilon_i \sim i.i.d.~WN(0, \sigma_{\epsilon_i}^2)~and~\displaystyle \sum_{i=1}^{\infty}\theta_i^2 < \infty
\end{align*}
\begin{align*}
Y_t &= \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} \\
&= \epsilon_t + \theta_1L\epsilon_t + \theta_2L^2\epsilon_t + \cdots + \theta_qL^q\epsilon_t \\
&= (1 + \theta_1L + \theta_2L^2 + \cdots + \theta_qL^q)\epsilon_t \\
&= \theta(L)\epsilon_t \\
\end{align*}
\begin{align*}
where~\epsilon_{t-1} = L\epsilon_t~and~\epsilon_{t-2} = L^2\epsilon_t$
\end{align*}

- **MA(1):**  
\begin{align*}
\text{Main Equation} && Y_t &= \epsilon_t + \theta_1\epsilon_{t-1} \\
\text{Expectation}   && E(Y_t) &= E(\epsilon_t + \theta_1\epsilon_{t-1}) = E(\epsilon_t) + \theta_1E(\epsilon_{t-1}) = 0 \\
\text{Variance}   && Var(Y_t) &= E[(\epsilon_t + \theta_1\epsilon_{t-1})^2] \\
&& &= E(\epsilon_t^2) + 2\theta_1E(\epsilon_{t}\epsilon_{t-1}) + \theta_1^2E(\epsilon_{t-1}^2) \\
&& &= \sigma_{\epsilon_i}^2 + 2 \theta_1 \cdot 0 + \theta_1^2 \sigma_{\epsilon_i}^2 \\
&& &= \sigma_{\epsilon_i}^2 + \theta_1^2\sigma_{\epsilon_i}^2 \\
\text{Covariance} && Cov(Y_t, Y_{t-1}) = \gamma_1 &= \text{E} \left[ (\epsilon_t + \theta_1 \epsilon_{t-1})(\epsilon_{t-1} + \theta_1 \epsilon_{t-2}) \right] \\
&& &= E (\epsilon_t \epsilon_{t-1}) + \theta_1 E (\epsilon_t \epsilon_{t-2}) + \theta_1 E (\epsilon_{t-1}^2) + \theta_1^2 E (\epsilon_{t-1} \epsilon_{t-2}) \\
&& &= 0 + \theta_1 \cdot 0 + \theta_1 \sigma_{\epsilon_{i}}^2 + \theta_1^2 \cdot 0 \\
&& &= \theta_1 \sigma_{\epsilon_{i}}^2   \\
&& Cov(Y_t, Y_{t-2}) = \gamma_2 &= \text{E} \left[ (\epsilon_t + \theta_1 \epsilon_{t-1})(\epsilon_{t-2} + \theta_1 \epsilon_{t-3}) \right] \\
&& &= E (\epsilon_t \epsilon_{t-2}) + \theta_1 E (\epsilon_t \epsilon_{t-3}) + \theta_1 E (\epsilon_{t-1} \epsilon_{t-2}) + \theta_1^2 E (\epsilon_{t-1} \epsilon_{t-3}) \\
&& &= 0 + \theta_1 \cdot 0 + \theta_1 \cdot 0 + \theta_1^2 \cdot 0 \\
&& &= 0 \\
\text{Autocorrelation} && Corr(Y_t, Y_{t-1}) = \rho_1 &= \dfrac{\theta_1}{1+\theta_1^2} \\
&& Corr(Y_t, Y_{t-i}) = \rho_i &= 0~~for~~i > 1 \\
\end{align*}

- **MA(2):**  
\begin{align*}
\text{Main Equation} && Y_t &= \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} \\
\text{Expectation}   && E(Y_t) &= E(\epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2}) = E(\epsilon_t) + \theta_1E(\epsilon_{t-1}) + \theta_2E(\epsilon_{t-2}) = 0 \\
\text{Variance}   && Var(Y_t) &= E[(\epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2})^2] \\
&& &= \sigma_{\epsilon_i}^2 + \theta_1^2\sigma_{\epsilon_i}^2 + \theta_2^2\sigma_{\epsilon_i}^2 \\
\text{Covariance} && Cov(Y_t, Y_{t-1}) = \gamma_1 &= \text{E} \left[ (\epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2\epsilon_{t-2})(\epsilon_{t-1} + \theta_1 \epsilon_{t-2} + \theta_2\epsilon_{t-3}) \right] \\
&& &= (\theta_1 + \theta_1\theta_2) \sigma_{\epsilon_{i}}^2   \\
&& Cov(Y_t, Y_{t-2}) = \gamma_2 &= \text{E} \left[ (\epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2\epsilon_{t-2})(\epsilon_{t-2} + \theta_1 \epsilon_{t-3} + \theta_2\epsilon_{t-4}) \right] \\
&& &= \theta_2 \sigma_{\epsilon_{i}}^2   \\
&& Cov(Y_t, Y_{t-i}) = \gamma_i &= 0~~for~~i > 2 \\
\text{Autocorrelation} && Corr(Y_t, Y_{t-1}) = \rho_1 &= \dfrac{\theta_1 + \theta_1 \theta_2}{1+\theta_1^2+\theta_2^2} \\
&& Corr(Y_t, Y_{t-2}) = \rho_2 &= \dfrac{\theta_2}{1+\theta_1^2+\theta_2^2} \\
&& Corr(Y_t, Y_{t-i}) = \rho_i &= 0~~for~~i > 2 \\
\end{align*}

- **MA(q):**  
\begin{align*}
\text{Autocorrelation} && Corr(Y_t, Y_{t-i}) = \rho_i &=
\begin{cases}
\dfrac{\theta_i + \theta_1\theta_{i-1}  + \theta_2\theta_{i-2} + \cdots + \theta_q\theta_{i-q}}{1 + \theta_1^2 + \cdots  + \theta_q^2} & \text{ for } i= 1, 2, \cdots, q \\
0 & \text{ for } i > q \\
\end{cases}
\end{align*}

> **움직임 특성:**
- **Stationarity Condition of MA(1):** $|\theta_1| < 1$
- **Stationarity Condition of MA(2):** $|\theta_2| < 1$, $\theta_1 + \theta_2 > -1$, $\theta_1 - \theta_2 < 1$


```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
```


```python
### MA(1)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([0.9])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_acf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_33_0](/assets/output_33_0.png)





![output_33_1](/assets/output_33_1.png)




```python
### MA(2)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([-1, 0.6])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_acf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_34_0](/assets/output_34_0.png)





![output_34_1](/assets/output_34_1.png)




```python
### MA(5)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([-1, 1.6, 0.9, -1.5, 0.7])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_acf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_35_0](/assets/output_35_0.png)





![output_35_1](/assets/output_35_1.png)



### AR(Auto-Regressive)

> **"$AR(p)$: 알고리즘의 차수($p$)가 유한한 자기자신의 과거값들의 선형조합"**
- **필요성:** ACF가 시차(Lag)가 증가해도 0이 되지 않고 오랜시간 남아있는 경우에 $MA$모형을 사용하면 차수가 $\infty$로 감

\begin{align*}
Y_t &= \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} + \epsilon_t \\
\end{align*}
\begin{align*}
where~\epsilon_i \sim i.i.d.~WN(0, \sigma_{\epsilon_i}^2)~and~\displaystyle \sum_{i=1}^{\infty}\phi_i^2 < \infty
\end{align*}
\begin{align*}
Y_t &= \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} + \epsilon_t \\
Y_t - \phi_1Y_{t-1} - \phi_2Y_{t-2} - \cdots - \phi_pY_{t-p} &= \epsilon_t \\
Y_t - \phi_1LY_t - \phi_2L^2Y_t - \cdots - \phi_pL^pY_t &= \epsilon_t \\
(1 - \phi_1L - \phi_2L^2 - \cdots - \phi_pL^p)Y_t &= \epsilon_t \\
\phi(L)Y_t &= \epsilon_t \\
\end{align*}
\begin{align*}
where~Y_{t-1} = LY_t~and~Y_{t-2} = L^2Y_t
\end{align*}

- **AR(1):**  
\begin{align*}
\text{Main Equation} && Y_t &= \phi_1 Y_{t-1} + \epsilon_t \\
&& &= \phi_1 (\phi_1 Y_{t-2} + \epsilon_{t-1}) + \epsilon_t \\
&& &= \phi_1^2 Y_{t-2} + \phi_1 \epsilon_{t-1} + \epsilon_t \\
&& &= \phi_1^2  (\phi_1 Y_{t-3} + \epsilon_{t-2}) + \phi_1 \epsilon_{t-1} + \epsilon_t \\
&& &= \phi_1^3 Y_{t-3} + \phi_1^2 \epsilon_{t-2} + \phi_1 \epsilon_{t-1} + \epsilon_t \\
&& & \vdots \\
&& &= \epsilon_t + \phi_1 \epsilon_{t-1} +\phi_1^2 \epsilon_{t-2} + \phi_1^3 \epsilon_{t-3} + \cdots  \\
&& &= MA(\infty) \\
\text{Expectation}   && E(Y_t) &= \mu = E(\phi_1 Y_{t-1} + \epsilon_t) = \phi_1 E(Y_{t-1}) + E(\epsilon_{t}) = \phi_1 \mu + 0 \\
&& (1-\phi_1)\mu &= 0 \\
&& \mu &= 0~~if~~\phi_1 \neq 1 \\
\text{Variance}   && Var(Y_t) &= \gamma_0 = E(Y_t^2) = E[(\phi_1 Y_{t-1} + \epsilon_t)^2] = E[ \phi_1^2  Y_{t-1}^2 + 2\phi_1 Y_{t-1} \epsilon_{t} + \epsilon_{t}^2] \\
&& &= \phi_1^2 E[ Y_{t-1}^2 ] + 2 \phi E[ Y_{t-1} \epsilon_{t} ] + E[ \epsilon_{t}^2 ] \\
&& &= \phi_1^2 \gamma_0 + 0 + \sigma_{\epsilon_i}^2 \\
&& (1-\phi_1^2)\gamma_0 &= \sigma_{\epsilon_i}^2 \\
&& \gamma_0 &= \dfrac{\sigma_{\epsilon_i}^2}{1-\phi_1^2}~~if~~\phi_1^2 \neq 1 \\
\text{Covariance} && Cov(Y_t, Y_{t-1}) &= \gamma_1 = E [(\phi_1 Y_{t-1} + \epsilon_t)(\phi_1 Y_{t-2} + \epsilon_{t-1})] \\
&& &= \phi_1^2E (Y_{t-1} Y_{t-2}) + \phi_1 E (Y_{t-1} \epsilon_{t-1}) + \phi_1 E (\epsilon_{t} Y_{t-2}) + E (\epsilon_{t} \epsilon_{t-1}) \\
&& &= \phi_1^2\gamma_1 + \phi_1 \sigma_{\epsilon_{i}}^2 + \phi_1 \cdot 0 + 0 \\
&& (1 - \phi_1^2)\gamma_1 &= \phi_1 \sigma_{\epsilon_{i}}^2 \\
&& \gamma_1 &= \dfrac{\phi_1 \sigma_{\epsilon_{i}}^2}{1 - \phi_1^2} \\
&& Cov(Y_t, Y_{t-2}) &= \gamma_2 = E [(\phi_1 Y_{t-1} + \epsilon_t)(\phi_1 Y_{t-3} + \epsilon_{t-2})] \\
&& &= \phi_1^2E (Y_{t-1} Y_{t-3}) + \phi_1 E (Y_{t-1} \epsilon_{t-2}) + \phi_1 E (\epsilon_{t} Y_{t-3}) + E (\epsilon_{t} \epsilon_{t-2}) \\
&& &= \phi_1^2\gamma_2 + \phi_1 E[(\phi_1Y_{t-2}+\epsilon_{t-1})\epsilon_{t-2}] + \phi_1 \cdot 0 + 0 \\
&& &= \phi_1^2\gamma_2 + \phi_1^2 E(Y_{t-2}\epsilon_{t-2}) + \phi_1 E(\epsilon_{t-1}\epsilon_{t-2}) \\
&& &= \phi_1^2\gamma_2 + \phi_1^2 \sigma_{\epsilon_{i}}^2 + 0 \\
&& (1 - \phi_1^2)\gamma_2 &= \phi_1^2 \sigma_{\epsilon_{i}}^2 \\
&& \gamma_2 &= \dfrac{\phi_1^2 \sigma_{\epsilon_{i}}^2}{1 - \phi_1^2} \\
\text{Autocorrelation} && Corr(Y_t, Y_{t-1}) = \rho_1 &= \phi_1 \\
&& Corr(Y_t, Y_{t-2}) = \rho_2 &= \phi_1^2 \\
&& Corr(Y_t, Y_{t-i}) = \rho_i &= \phi_1^i \\
\end{align*}

> **움직임 특성:**
- $\phi_1 = 0$: $Y_t$는 백색잡음  
- $\phi_1 < 0$: 부호를 바꿔가면서(진동하면서) 지수적으로 감소  
- $\phi_1 > 0$: 시차가 증가하면서 자기상관계수는 지수적으로 감소  
- $\phi_1 = 1$: $Y_t$는 비정상성인 랜덤워크(Random Walk)
\begin{align*}
Y_t &= Y_{t-1} + \epsilon_t \\
Var(Y_t) &= Var(Y_{t-1} + \epsilon_t) \\
&= Var(Y_{t-1}) + Var(\epsilon_t)  \;\; (\text{independence}) \\
Var(Y_t) &> Var(Y_{t-1})
\end{align*}
- **Stationarity Condition:** $|\phi_1| < 1$

- **AR(2):**  
\begin{align*}
\text{Main Equation} && Y_t &= \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t \\
&& &= \phi_1 (\phi_1 Y_{t-2} + \epsilon_{t-1}) + \phi_2 (\phi_2 Y_{t-3} + \epsilon_{t-2}) + \epsilon_t \\
&& &= \phi_1^2 Y_{t-2} + \phi_1 \epsilon_{t-1} + \phi_2^2 Y_{t-3} + \phi_2 \epsilon_{t-2} + \epsilon_t \\
&& &= \phi_1^2 (\phi_1 Y_{t-3} + \phi_2 Y_{t-4} + \epsilon_{t-3}) + \phi_1 \epsilon_{t-1} +
\phi_2^2 (\phi_1 Y_{t-4} + \phi_2 Y_{t-5} + \epsilon_{t-4}) + \phi_2 \epsilon_{t-2} + \epsilon_t \\
&& & \vdots \\
&& &= \epsilon_t + \phi_1 \epsilon_{t-1} + \phi_2 \epsilon_{t-1} + \phi_1^2 \epsilon_{t-2} + \phi_2^2 \epsilon_{t-2} + \phi_1^3 \epsilon_{t-3} + \phi_2^3 \epsilon_{t-3} + \cdots  \\
&& &= MA(\infty) \\
\text{Expectation}   && E(Y_t) &= \mu = E(\phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t) = \phi_1 E(Y_{t-1}) + \phi_2 E(Y_{t-2}) + E(\epsilon_{t}) = \phi_1 \mu + \phi_2 \mu + 0 \\
&& (1-\phi_1-\phi_2)\mu &= 0 \\
&& \mu &= 0~~if~~\phi_1+\phi_2 \neq 1 \\
\text{Covariance("Yule-Walker Equation")} && \gamma_i &= E(Y_tY_{t-i}) = E[(\phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t)Y_{t-i}] \\
&& &= E(\phi_1 Y_{t-1}Y_{t-i}) + E(\phi_2 Y_{t-2}Y_{t-i}) + E(\epsilon_t Y_{t-i}) \\
&& &= \phi_1 \gamma_{i-1} + \phi_2 \gamma_{i-2} \\
\text{Autocorrelation} && Corr(Y_t, Y_{t-i}) &= \rho_i = \phi_1 \rho_{i-1} + \phi_2 \rho_{i-2} \\
&& \rho_1 &= \phi_1 \rho_{0} + \phi_2 \rho_{-1} = \phi_1 \cdot 1 + \phi_2 \rho_{1} \\
&& (1-\phi_2)\rho_1 &= \phi_1 \\
&& \rho_1 &= \dfrac{\phi_1}{1-\phi_2} \\
&& & \vdots \\
&& \rho_2 &= \dfrac{\phi_1^2 + \phi_2(1-\phi_2)}{1-\phi_2} \\
&& & \vdots \\
&& \rho_i &= \left( 1+\dfrac{1+\phi_2}{1-\phi_2} \cdot i \right)\left(\dfrac{\phi_1}{2} \right)^i \\
\end{align*}

> **움직임 특성:**
- 시차가 증가하면서 자기상관계수의 절대값은 지수적으로 감소  
- 진동 주파수에 따라 다르지만 진동 가능
- **Stationarity Condition:** $|\phi_1| < 1$, $\phi_1 + \phi_2 < 1$, $\phi_2 - \phi_1 < 1$


```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
```


```python
### AR(1)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0.9])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling partial autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_38_0](/assets/output_38_0.png)



    <ipython-input-2-27406e332a5a>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
    <ipython-input-2-27406e332a5a>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))




![output_38_2](/assets/output_38_2.png)




```python
### AR(1)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([-0.9])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling partial autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_39_0](/assets/output_39_0.png)



    <ipython-input-3-b6641c9838c9>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
    <ipython-input-3-b6641c9838c9>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))




![output_39_2](/assets/output_39_2.png)




```python
### AR(2)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0.5, 0.25])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling partial autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_40_0](/assets/output_40_0.png)



    <ipython-input-4-f17197db6fc1>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
    <ipython-input-4-f17197db6fc1>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))




![output_40_2](/assets/output_40_2.png)




```python
### AR(2)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([-0.5, 0.25])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling partial autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_41_0](/assets/output_41_0.png)



    <ipython-input-5-d1825d0b388d>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
    <ipython-input-5-d1825d0b388d>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))




![output_41_2](/assets/output_41_2.png)




```python
### AR(5)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0.5, 0.25, -0.3, 0.1, -0.1])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function of an ARMA process")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical partial autocorrelation function of an ARMA process")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling partial autocorrelation function of an ARMA process")
plt.tight_layout()
plt.show()
```



![output_42_0](/assets/output_42_0.png)



    <ipython-input-6-c23900944ac7>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=10))
    <ipython-input-6-c23900944ac7>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=10))




![output_42_2](/assets/output_42_2.png)



### Relation of MA and AR

- **가역성 조건(Invertibility Condition):**

> **1)** $MA(q)$ -> $AR(\infty)$: 변환 후 AR 모형이 Stationary Condition이면 "Invertibility"  
> **2)** $AR(p)$ -> $MA(\infty)$: 여러개 모형변환 가능하지만 "Invertibility" 조건을 만족하는 MA 모형은 단 1개만 존재

### ARMA(Auto-Regressive Moving Average)

> **"$ARMA(p,q)$: 알고리즘의 차수($p~and~q$)가 유한한 $AR(p)$와 $MA(q)$의 선형조합"**  
> - $AR$과 $MA$의 정상성 조건과 가역성 조건이 동일하게 $ARMA$ 알고리즘들에 적용  
> - 종속변수 $Y_t$는 종속변수 $Y_t$와 백색잡음($\epsilon_t$) 차분들(Lagged Variables)의 합으로 예측가능  

\begin{align*}
Y_t = \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} +
\theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} + \epsilon_t \\
\end{align*}
\begin{align*}
where~\epsilon_i \sim i.i.d.~WN(0, \sigma_{\epsilon_i}^2)~and~\displaystyle \sum_{i=1}^{\infty}\phi_i^2 < \infty, \displaystyle \sum_{i=1}^{\infty}\theta_i^2 < \infty
\end{align*}
\begin{align*}
\phi(L)Y_t &= \theta(L)\epsilon_t \\
Y_t &= \dfrac{\theta(L)}{\phi(L)}\epsilon_t \\
\end{align*}

\begin{align*}
\text{Main Equation} && Y_t &= \dfrac{\theta(L)}{\phi(L)}\epsilon_t \\
&& &= \psi(L)\epsilon_t \text{ where } \psi(L) = \dfrac{\theta(L)}{\phi(L)} \\
&& &= (1 + \psi_1L + \psi_2L^2 + \cdots)\epsilon_t \\
&& &= \epsilon_t + \psi_1\epsilon_{t-1} + \psi_2\epsilon_{t-2} + \cdots \\
&& & \text{ where } \\
&& \psi_1 &= \theta_1 - \phi_1 \\
&& \psi_2 &= \theta_2 - \phi_2 - \phi_1 \psi_1 \\
&& & \vdots \\
&& \psi_j &= \theta_j - \phi_p\psi_{j-p} - \phi_2 \psi_{p-1} - \cdots - \phi_1 \psi_{j-1} \\
\text{Autocorrelation("Yule-Walker Equation")} && \rho_i &= \phi_1 \rho_{i-1} + \cdots + \phi_p \rho_{i-p} \\
\end{align*}


```python
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
```


```python
### ARMA(1,0) = AR(1)
# Setting
np.random.seed(123)
ar_params = np.array([0.75])
ma_params = np.array([])
index_name = ['const', 'ar(1)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2775.2536</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.3</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-07-31 22:45</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>2</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>998</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>4.0000</td>              <td>HQIC:</td>        <td>2766.126</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2760.5304</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th> <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0406</td>  <td>0.1197</td>  <td>0.3389</td>  <td>0.7347</td> <td>-0.1940</td> <td>0.2752</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>0.7475</td>  <td>0.0210</td>  <td>35.6193</td> <td>0.0000</td> <td>0.7063</td>  <td>0.7886</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>  <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>1.3379</td>  <td>0.0000</td>   <td>1.3379</td>   <td>0.0000</td>  
</tr>
</table>




![output_46_1](/assets/output_46_1.png)





![output_46_2](/assets/output_46_2.png)




```python
### ARMA(2,0) = AR(2)
# Setting
np.random.seed(123)
ar_params = np.array([0.75, -0.25])
ma_params = np.array([])
index_name = ['const', 'ar(1)', 'ar(2)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2781.5944</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.0</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:30</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>997</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>5.0000</td>              <td>HQIC:</td>        <td>2769.424</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2761.9633</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0193</td>   <td>0.0592</td>  <td>0.3255</td>  <td>0.7448</td> <td>-0.0967</td> <td>0.1352</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>0.7557</td>   <td>0.0305</td>  <td>24.7647</td> <td>0.0000</td> <td>0.6959</td>  <td>0.8155</td>
</tr>
<tr>
  <th>ar.L2.y</th> <td>-0.2678</td>  <td>0.0305</td>  <td>-8.7791</td> <td>0.0000</td> <td>-0.3276</td> <td>-0.2080</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>  <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>1.4111</td>  <td>-1.3203</td>  <td>1.9324</td>   <td>-0.1197</td>
</tr>
<tr>
  <th>AR.2</th> <td>1.4111</td>  <td>1.3203</td>   <td>1.9324</td>   <td>0.1197</td>  
</tr>
</table>




![output_47_1](/assets/output_47_1.png)





![output_47_2](/assets/output_47_2.png)




```python
### ARMA(4,0) = AR(4)
# Setting
np.random.seed(123)
ar_params = np.array([0.75, -0.25, 0.2, -0.5])
ma_params = np.array([])
index_name = ['const', 'ar(1)', 'ar(2)', 'ar(3)', 'ar(4)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2796.4388</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.5</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:32</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>995</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.958</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>12.0000</td>             <td>HQIC:</td>        <td>2778.184</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2766.9923</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0093</td>   <td>0.0374</td>   <td>0.2500</td>  <td>0.8026</td> <td>-0.0639</td> <td>0.0826</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>0.7470</td>   <td>0.0279</td>   <td>26.7627</td> <td>0.0000</td> <td>0.6923</td>  <td>0.8017</td>
</tr>
<tr>
  <th>ar.L2.y</th> <td>-0.2521</td>  <td>0.0362</td>   <td>-6.9600</td> <td>0.0000</td> <td>-0.3231</td> <td>-0.1811</td>
</tr>
<tr>
  <th>ar.L3.y</th> <td>0.1631</td>   <td>0.0362</td>   <td>4.4995</td>  <td>0.0000</td> <td>0.0920</td>  <td>0.2341</td>
</tr>
<tr>
  <th>ar.L4.y</th> <td>-0.4699</td>  <td>0.0279</td>  <td>-16.8256</td> <td>0.0000</td> <td>-0.5247</td> <td>-0.4152</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>0.8756</td>   <td>-0.6261</td>  <td>1.0764</td>   <td>-0.0988</td>
</tr>
<tr>
  <th>AR.2</th> <td>0.8756</td>   <td>0.6261</td>   <td>1.0764</td>   <td>0.0988</td>  
</tr>
<tr>
  <th>AR.3</th> <td>-0.7021</td>  <td>-1.1592</td>  <td>1.3553</td>   <td>-0.3367</td>
</tr>
<tr>
  <th>AR.4</th> <td>-0.7021</td>  <td>1.1592</td>   <td>1.3553</td>   <td>0.3367</td>  
</tr>
</table>




![output_48_1](/assets/output_48_1.png)





![output_48_2](/assets/output_48_2.png)




```python
### ARMA(0,1) = MA(1)
# Setting
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([0.65])
index_name = ['const', 'ma(1)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\base\model.py:567: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warn("Maximum Likelihood optimization failed to converge. "



<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2775.2511</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.3</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:33</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>2</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>998</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>0.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>3.0000</td>              <td>HQIC:</td>        <td>2766.124</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2760.5278</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th> <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0157</td>  <td>0.0500</td>  <td>0.3146</td>  <td>0.7531</td> <td>-0.0823</td> <td>0.1138</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>0.6508</td>  <td>0.0246</td>  <td>26.5050</td> <td>0.0000</td> <td>0.6027</td>  <td>0.6989</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>MA.1</th> <td>-1.5365</td>  <td>0.0000</td>   <td>1.5365</td>   <td>0.5000</td>  
</tr>
</table>




![output_49_2](/assets/output_49_2.png)





![output_49_3](/assets/output_49_3.png)



```python
### ARMA(0,2) = MA(2)
# Setting
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([0.65, -0.25])
index_name = ['const', 'ma(1)', 'ma(2)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2783.1398</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.8</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:34</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>997</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>8.0000</td>              <td>HQIC:</td>        <td>2770.970</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2763.5088</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0130</td>   <td>0.0425</td>  <td>0.3053</td>  <td>0.7602</td> <td>-0.0703</td> <td>0.0962</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>0.6501</td>   <td>0.0310</td>  <td>20.9416</td> <td>0.0000</td> <td>0.5892</td>  <td>0.7109</td>
</tr>
<tr>
  <th>ma.L2.y</th> <td>-0.2487</td>  <td>0.0307</td>  <td>-8.0906</td> <td>0.0000</td> <td>-0.3090</td> <td>-0.1885</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>MA.1</th> <td>-1.0865</td>  <td>0.0000</td>   <td>1.0865</td>   <td>0.5000</td>  
</tr>
<tr>
  <th>MA.2</th> <td>3.7001</td>   <td>0.0000</td>   <td>3.7001</td>   <td>0.0000</td>  
</tr>
</table>




![output_50_1](/assets/output_50_1.png)





![output_50_2](/assets/output_50_2.png)




```python
### ARMA(0,4) = MA(4)
# Setting
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([0.65, -0.25, 0.5, -0.9])
index_name = ['const', 'ma(1)', 'ma(2)', 'ma(3)', 'ma(4)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>3472.2476</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1715.4</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:36</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>995</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>1.344</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>12.0000</td>             <td>HQIC:</td>        <td>3453.993</td>
</tr>
<tr>
         <td>AIC:</td>             <td>3442.8011</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0079</td>   <td>0.0297</td>   <td>0.2653</td>  <td>0.7908</td> <td>-0.0504</td> <td>0.0662</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>-0.0418</td>  <td>0.0275</td>   <td>-1.5186</td> <td>0.1289</td> <td>-0.0957</td> <td>0.0121</td>
</tr>
<tr>
  <th>ma.L2.y</th> <td>0.2823</td>   <td>0.0276</td>   <td>10.2186</td> <td>0.0000</td> <td>0.2282</td>  <td>0.3365</td>
</tr>
<tr>
  <th>ma.L3.y</th> <td>-0.0569</td>  <td>0.0273</td>   <td>-2.0805</td> <td>0.0375</td> <td>-0.1105</td> <td>-0.0033</td>
</tr>
<tr>
  <th>ma.L4.y</th> <td>-0.4858</td>  <td>0.0279</td>  <td>-17.4311</td> <td>0.0000</td> <td>-0.5404</td> <td>-0.4311</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>MA.1</th> <td>1.2761</td>   <td>-0.0000</td>  <td>1.2761</td>   <td>-0.0000</td>
</tr>
<tr>
  <th>MA.2</th> <td>-0.0088</td>  <td>-1.0828</td>  <td>1.0829</td>   <td>-0.2513</td>
</tr>
<tr>
  <th>MA.3</th> <td>-0.0088</td>  <td>1.0828</td>   <td>1.0829</td>   <td>0.2513</td>  
</tr>
<tr>
  <th>MA.4</th> <td>-1.3757</td>  <td>-0.0000</td>  <td>1.3757</td>   <td>-0.5000</td>
</tr>
</table>




![output_51_1](/assets/output_51_1.png)





![output_51_2](/assets/output_51_2.png)




```python
### ARMA(1,1)
# Setting
np.random.seed(123)
ar_params = np.array([0.75])
ma_params = np.array([0.65])
index_name = ['const', 'ar(1)', 'ma(1)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2783.7601</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1378.1</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-07-31 22:45</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>997</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>9.0000</td>              <td>HQIC:</td>        <td>2771.590</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2764.1291</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th> <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0641</td>  <td>0.1970</td>  <td>0.3252</td>  <td>0.7450</td> <td>-0.3220</td> <td>0.4501</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>0.7465</td>  <td>0.0224</td>  <td>33.3502</td> <td>0.0000</td> <td>0.7027</td>  <td>0.7904</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>0.6519</td>  <td>0.0261</td>  <td>24.9637</td> <td>0.0000</td> <td>0.6007</td>  <td>0.7031</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>1.3395</td>   <td>0.0000</td>   <td>1.3395</td>   <td>0.0000</td>  
</tr>
<tr>
  <th>MA.1</th> <td>-1.5340</td>  <td>0.0000</td>   <td>1.5340</td>   <td>0.5000</td>  
</tr>
</table>




![output_52_1](/assets/output_52_1.png)


![output_52_2](/assets/output_52_2.png)


```python
### ARMA(2,2)
# Setting
np.random.seed(123)
ar_params = np.array([0.75, -0.25])
ma_params = np.array([0.65, 0.5])
index_name = ['const', 'ar(1)', 'ar(2)', 'ma(1)', 'ma(2)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2796.4977</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1377.5</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:40</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>995</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.958</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>12.0000</td>             <td>HQIC:</td>        <td>2778.243</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2767.0512</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0415</td>   <td>0.1265</td>  <td>0.3283</td>  <td>0.7427</td> <td>-0.2065</td> <td>0.2895</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>0.7472</td>   <td>0.0597</td>  <td>12.5090</td> <td>0.0000</td> <td>0.6301</td>  <td>0.8643</td>
</tr>
<tr>
  <th>ar.L2.y</th> <td>-0.2675</td>  <td>0.0518</td>  <td>-5.1626</td> <td>0.0000</td> <td>-0.3691</td> <td>-0.1660</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>0.6544</td>   <td>0.0553</td>  <td>11.8393</td> <td>0.0000</td> <td>0.5460</td>  <td>0.7627</td>
</tr>
<tr>
  <th>ma.L2.y</th> <td>0.5209</td>   <td>0.0384</td>  <td>13.5542</td> <td>0.0000</td> <td>0.4456</td>  <td>0.5963</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>1.3966</td>   <td>-1.3371</td>  <td>1.9334</td>   <td>-0.1215</td>
</tr>
<tr>
  <th>AR.2</th> <td>1.3966</td>   <td>1.3371</td>   <td>1.9334</td>   <td>0.1215</td>  
</tr>
<tr>
  <th>MA.1</th> <td>-0.6281</td>  <td>-1.2350</td>  <td>1.3855</td>   <td>-0.3249</td>
</tr>
<tr>
  <th>MA.2</th> <td>-0.6281</td>  <td>1.2350</td>   <td>1.3855</td>   <td>0.3249</td>  
</tr>
</table>




![output_53_1](/assets/output_53_1.png)





![output_53_2](/assets/output_53_2.png)




```python
### ARMA(5,5)
# Setting
np.random.seed(123)
ar_params = np.array([0.75, -0.25, 0.5, -0.5, -0.1])
ma_params = np.array([0.65, 0.5, 0.2, -0.5, -0.1])
index_name = ['const', 'ar(1)', 'ar(2)', 'ar(3)', 'ar(4)', 'ar(5)', 'ma(1)', 'ma(2)', 'ma(3)', 'ma(4)', 'ma(5)']
ahead = 100
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order = len(ar)-1, len(ma)-1

# Generator
y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit = statsmodels.tsa.arima_model.ARMA(y, (ar_order,ma_order)).fit(trend='c', disp=0)
pred_ts_point = fit.forecast(steps=ahead)[0]
pred_ts_interval = fit.forecast(steps=ahead)[2]
ax = pd.DataFrame(y).plot(figsize=(12,5))
forecast_index = [i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15)
plt.legend(['observed', 'forecast'])
display(fit.summary2())
plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()

# ACF가 오히려 증가되는 이유는?
```


<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARMA</td>               <td>BIC:</td>         <td>2844.4865</td>
</tr>
<tr>
  <td>Dependent Variable:</td>         <td>y</td>           <td>Log-Likelihood:</td>    <td>-1380.8</td>
</tr>
<tr>
         <td>Date:</td>        <td>2020-09-29 23:41</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>        <td>1000</td>              <td>Method:</td>        <td>css-mle</td>
</tr>
<tr>
       <td>Df Model:</td>             <td>11</td>               <td>Sample:</td>           <td>0</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>989</td>                 <td></td>               <td>0</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>   <td>0.959</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>54.0000</td>             <td>HQIC:</td>        <td>2807.977</td>
</tr>
<tr>
         <td>AIC:</td>             <td>2785.5934</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>      <th>Coef.</th>  <th>Std.Err.</th>    <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th>
</tr>
<tr>
  <th>const</th>   <td>0.0308</td>   <td>0.0864</td>  <td>0.3569</td>  <td>0.7212</td> <td>-0.1386</td> <td>0.2003</td>
</tr>
<tr>
  <th>ar.L1.y</th> <td>1.3387</td>   <td>0.4949</td>  <td>2.7049</td>  <td>0.0068</td> <td>0.3687</td>  <td>2.3087</td>
</tr>
<tr>
  <th>ar.L2.y</th> <td>-0.7833</td>  <td>0.4780</td>  <td>-1.6387</td> <td>0.1013</td> <td>-1.7202</td> <td>0.1536</td>
</tr>
<tr>
  <th>ar.L3.y</th> <td>0.7138</td>   <td>0.2246</td>  <td>3.1786</td>  <td>0.0015</td> <td>0.2737</td>  <td>1.1540</td>
</tr>
<tr>
  <th>ar.L4.y</th> <td>-0.7757</td>  <td>0.2729</td>  <td>-2.8428</td> <td>0.0045</td> <td>-1.3106</td> <td>-0.2409</td>
</tr>
<tr>
  <th>ar.L5.y</th> <td>0.2339</td>   <td>0.2687</td>  <td>0.8705</td>  <td>0.3840</td> <td>-0.2927</td> <td>0.7604</td>
</tr>
<tr>
  <th>ma.L1.y</th> <td>0.0678</td>   <td>0.4967</td>  <td>0.1366</td>  <td>0.8914</td> <td>-0.9057</td> <td>1.0413</td>
</tr>
<tr>
  <th>ma.L2.y</th> <td>0.2093</td>   <td>0.2164</td>  <td>0.9674</td>  <td>0.3334</td> <td>-0.2148</td> <td>0.6334</td>
</tr>
<tr>
  <th>ma.L3.y</th> <td>-0.0514</td>  <td>0.1982</td>  <td>-0.2592</td> <td>0.7955</td> <td>-0.4398</td> <td>0.3371</td>
</tr>
<tr>
  <th>ma.L4.y</th> <td>-0.6159</td>  <td>0.0713</td>  <td>-8.6358</td> <td>0.0000</td> <td>-0.7557</td> <td>-0.4762</td>
</tr>
<tr>
  <th>ma.L5.y</th> <td>0.1658</td>   <td>0.2752</td>  <td>0.6026</td>  <td>0.5467</td> <td>-0.3735</td> <td>0.7052</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>-0.4994</td>  <td>-1.1024</td>  <td>1.2103</td>   <td>-0.3177</td>
</tr>
<tr>
  <th>AR.2</th> <td>-0.4994</td>  <td>1.1024</td>   <td>1.2103</td>   <td>0.3177</td>  
</tr>
<tr>
  <th>AR.3</th> <td>1.0036</td>   <td>-0.5072</td>  <td>1.1245</td>   <td>-0.0745</td>
</tr>
<tr>
  <th>AR.4</th> <td>1.0036</td>   <td>0.5072</td>   <td>1.1245</td>   <td>0.0745</td>  
</tr>
<tr>
  <th>AR.5</th> <td>2.3086</td>   <td>-0.0000</td>  <td>2.3086</td>   <td>-0.0000</td>
</tr>
<tr>
  <th>MA.1</th> <td>-1.1194</td>  <td>-0.0000</td>  <td>1.1194</td>   <td>-0.5000</td>
</tr>
<tr>
  <th>MA.2</th> <td>-0.0971</td>  <td>-1.0335</td>  <td>1.0381</td>   <td>-0.2649</td>
</tr>
<tr>
  <th>MA.3</th> <td>-0.0971</td>  <td>1.0335</td>   <td>1.0381</td>   <td>0.2649</td>  
</tr>
<tr>
  <th>MA.4</th> <td>1.3648</td>   <td>-0.0000</td>  <td>1.3648</td>   <td>-0.0000</td>
</tr>
<tr>
  <th>MA.5</th> <td>3.6628</td>   <td>-0.0000</td>  <td>3.6628</td>   <td>-0.0000</td>
</tr>
</table>




![output_54_1](/assets/output_54_1.png)





![output_54_2](/assets/output_54_2.png)



### 모형 차수결정 정리

> **1) 추정 및 예측을 하기 전에 파라미터에 따라 모형이 어떠한 결과를 도출할지 이해(예상) 필요**  
> **2) 결과이해(예상)는 기계의 실수를 방지하고 결과의 확신을 증가시킴**  

- **$p$, $q$ 파라미터 추론(by ACF and PACF):**  
1) 정상성 형태 변환: 차분/로그변환/계절성제거 등을 통해 데이터를 정상성 형태로 변환  
2) $ACF$, $PACF$를 도식화 하여 ARMA의 파라미터 차수를 추론  

| - | 자기회귀: $AR(p)$ | 이동평균: $MA(q)$ | 자기회귀이동평균: $ARMA(p,q)$ |
|----------------------|-------------------------------------------|-------------------------------------------|--------------------------------------------------------------|
| $ACF$ | 지수적 감소, 진동하는 사인 형태 | $q+1$ 차항부터 절단모양(0수렴) | $q+1$ 차항부터 지수적 감소 혹은 진동하는 사인형태 |
| $PACF$ | $p+1$ 차항부터 절단모양(0수렴) | 지수적 감소, 진동하는 사인 형태 | $p+1$ 차항부터 지수적 감소 혹은 진동하는 사인형태 |

### 실습: 항체형성 호르몬수지 ARMA 모델링


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터로딩 및 확인
data = sm.datasets.get_rdataset("lh")
raw = data.data
raw.plot(x='time', y='value')
plt.show()

# ACF/PACF 확인
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(raw.value, lags=10, ax=plt.subplot(211))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("ACF")

sm.graphics.tsa.plot_pacf(raw.value, lags=10, ax=plt.subplot(212))
plt.xlim(-1, 11)
plt.ylim(-1.1, 1.1)
plt.title("PACF")
plt.tight_layout()
plt.show()

# MA(1) 모델링
fit = sm.tsa.ARMA(raw.value, (0,1)).fit()
display(fit.summary())

# AR(1) 모델링
fit = sm.tsa.ARMA(raw.value, (1,0)).fit()
display(fit.summary())

# ARMA(1,1) 모델링
fit = sm.tsa.ARMA(raw.value, (1,1)).fit()
display(fit.summary())
```



![output_57_0](/assets/output_57_0.png)





![output_57_1](/assets/output_57_1.png)




<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>value</td>      <th>  No. Observations:  </th>   <td>48</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(0, 1)</td>    <th>  Log Likelihood     </th> <td>-31.052</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>0.461</td>
</tr>
<tr>
  <th>Date:</th>          <td>Tue, 29 Sep 2020</td> <th>  AIC                </th> <td>68.104</td>
</tr>
<tr>
  <th>Time:</th>              <td>23:54:06</td>     <th>  BIC                </th> <td>73.717</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>70.225</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    2.4050</td> <td>    0.098</td> <td>   24.576</td> <td> 0.000</td> <td>    2.213</td> <td>    2.597</td>
</tr>
<tr>
  <th>ma.L1.value</th> <td>    0.4810</td> <td>    0.094</td> <td>    5.093</td> <td> 0.000</td> <td>    0.296</td> <td>    0.666</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>MA.1</th> <td>          -2.0790</td> <td>          +0.0000j</td> <td>           2.0790</td> <td>           0.5000</td>
</tr>
</table>



<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>value</td>      <th>  No. Observations:  </th>   <td>48</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(1, 0)</td>    <th>  Log Likelihood     </th> <td>-29.379</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>0.444</td>
</tr>
<tr>
  <th>Date:</th>          <td>Tue, 29 Sep 2020</td> <th>  AIC                </th> <td>64.758</td>
</tr>
<tr>
  <th>Time:</th>              <td>23:54:06</td>     <th>  BIC                </th> <td>70.372</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>66.880</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    2.4133</td> <td>    0.147</td> <td>   16.460</td> <td> 0.000</td> <td>    2.126</td> <td>    2.701</td>
</tr>
<tr>
  <th>ar.L1.value</th> <td>    0.5739</td> <td>    0.116</td> <td>    4.939</td> <td> 0.000</td> <td>    0.346</td> <td>    0.802</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.7424</td> <td>          +0.0000j</td> <td>           1.7424</td> <td>           0.0000</td>
</tr>
</table>



<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>value</td>      <th>  No. Observations:  </th>   <td>48</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(1, 1)</td>    <th>  Log Likelihood     </th> <td>-28.762</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>0.439</td>
</tr>
<tr>
  <th>Date:</th>          <td>Tue, 29 Sep 2020</td> <th>  AIC                </th> <td>65.524</td>
</tr>
<tr>
  <th>Time:</th>              <td>23:54:06</td>     <th>  BIC                </th> <td>73.009</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>68.353</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    2.4101</td> <td>    0.136</td> <td>   17.754</td> <td> 0.000</td> <td>    2.144</td> <td>    2.676</td>
</tr>
<tr>
  <th>ar.L1.value</th> <td>    0.4522</td> <td>    0.177</td> <td>    2.556</td> <td> 0.011</td> <td>    0.105</td> <td>    0.799</td>
</tr>
<tr>
  <th>ma.L1.value</th> <td>    0.1982</td> <td>    0.171</td> <td>    1.162</td> <td> 0.245</td> <td>   -0.136</td> <td>    0.532</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           2.2114</td> <td>          +0.0000j</td> <td>           2.2114</td> <td>           0.0000</td>
</tr>
<tr>
  <th>MA.1</th> <td>          -5.0462</td> <td>          +0.0000j</td> <td>           5.0462</td> <td>           0.5000</td>
</tr>
</table>


### 실습: 호흡기질환 사망자수 ARMA 모델링


```python
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터로딩 및 확인
data = sm.datasets.get_rdataset("deaths", "MASS")
raw = data.data
raw.value = np.log(raw.value)
raw.plot(x='time', y='value')
plt.show()

# ACF/PACF 확인
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(raw.value.values, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("ACF")

sm.graphics.tsa.plot_pacf(raw.value.values, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("PACF")
plt.tight_layout()
plt.show()

# ARMA(1,1) 모델링
fit = sm.tsa.ARMA(raw.value, (1,1)).fit()
display(fit.summary())
```



![output_59_0](/assets/output_59_0.png)





![output_59_1](/assets/output_59_1.png)




<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>value</td>      <th>  No. Observations:  </th>   <td>72</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(1, 1)</td>    <th>  Log Likelihood     </th> <td>31.983</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>0.154</td>
</tr>
<tr>
  <th>Date:</th>          <td>Tue, 29 Sep 2020</td> <th>  AIC                </th> <td>-55.965</td>
</tr>
<tr>
  <th>Time:</th>              <td>23:58:22</td>     <th>  BIC                </th> <td>-46.859</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>-52.340</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    7.6037</td> <td>    0.080</td> <td>   94.470</td> <td> 0.000</td> <td>    7.446</td> <td>    7.761</td>
</tr>
<tr>
  <th>ar.L1.value</th> <td>    0.6796</td> <td>    0.098</td> <td>    6.970</td> <td> 0.000</td> <td>    0.489</td> <td>    0.871</td>
</tr>
<tr>
  <th>ma.L1.value</th> <td>    0.4680</td> <td>    0.111</td> <td>    4.214</td> <td> 0.000</td> <td>    0.250</td> <td>    0.686</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.4713</td> <td>          +0.0000j</td> <td>           1.4713</td> <td>           0.0000</td>
</tr>
<tr>
  <th>MA.1</th> <td>          -2.1369</td> <td>          +0.0000j</td> <td>           2.1369</td> <td>           0.5000</td>
</tr>
</table>



```python
from itertools import product

# ARMA(p,q) 모델링
result = []
for p, q in product(range(4), range(2)):
    if (p == 0 & q == 0):
        continue
    model = sm.tsa.ARMA(raw.value, (p, q)).fit()
    try:
        result.append({"p": p, "q": q, "LLF": model.llf, "AIC": model.aic, "BIC": model.bic})
    except:
        pass

# 모형 최적모수 선택
result = pd.DataFrame(result)
display(result)
opt_ar = result.iloc[np.argmin(result['AIC']), 0]
opt_ma = result.iloc[np.argmin(result['AIC']), 1]

# ARMA 모델링
fit = sm.tsa.ARMA(raw.value, (opt_ar,opt_ma)).fit()
display(fit.summary())

# 잔차 ACF/PACF 확인
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(fit.resid, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("ACF")

sm.graphics.tsa.plot_pacf(fit.resid, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("PACF")
plt.tight_layout()
plt.show()
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
      <th>p</th>
      <th>q</th>
      <th>LLF</th>
      <th>AIC</th>
      <th>BIC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>24.894297</td>
      <td>-43.788594</td>
      <td>-36.958595</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>31.982602</td>
      <td>-55.965203</td>
      <td>-46.858539</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>35.739122</td>
      <td>-63.478243</td>
      <td>-54.371579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>44.512880</td>
      <td>-79.025760</td>
      <td>-67.642430</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>38.560200</td>
      <td>-67.120400</td>
      <td>-55.737069</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>1</td>
      <td>45.279957</td>
      <td>-78.559913</td>
      <td>-64.899917</td>
    </tr>
  </tbody>
</table>
</div>



<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>value</td>      <th>  No. Observations:  </th>   <td>72</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(2, 1)</td>    <th>  Log Likelihood     </th> <td>44.513</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>0.128</td>
</tr>
<tr>
  <th>Date:</th>          <td>Wed, 30 Sep 2020</td> <th>  AIC                </th> <td>-79.026</td>
</tr>
<tr>
  <th>Time:</th>              <td>00:02:54</td>     <th>  BIC                </th> <td>-67.642</td>
</tr>
<tr>
  <th>Sample:</th>                <td>0</td>        <th>  HQIC               </th> <td>-74.494</td>
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>    7.5920</td> <td>    0.020</td> <td>  384.993</td> <td> 0.000</td> <td>    7.553</td> <td>    7.631</td>
</tr>
<tr>
  <th>ar.L1.value</th> <td>    1.6407</td> <td>    0.059</td> <td>   27.830</td> <td> 0.000</td> <td>    1.525</td> <td>    1.756</td>
</tr>
<tr>
  <th>ar.L2.value</th> <td>   -0.8787</td> <td>    0.055</td> <td>  -16.027</td> <td> 0.000</td> <td>   -0.986</td> <td>   -0.771</td>
</tr>
<tr>
  <th>ma.L1.value</th> <td>   -0.7021</td> <td>    0.075</td> <td>   -9.363</td> <td> 0.000</td> <td>   -0.849</td> <td>   -0.555</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           0.9336</td> <td>          -0.5162j</td> <td>           1.0668</td> <td>          -0.0804</td>
</tr>
<tr>
  <th>AR.2</th> <td>           0.9336</td> <td>          +0.5162j</td> <td>           1.0668</td> <td>           0.0804</td>
</tr>
<tr>
  <th>MA.1</th> <td>           1.4243</td> <td>          +0.0000j</td> <td>           1.4243</td> <td>           0.0000</td>
</tr>
</table>




![output_60_2](/assets/output_60_2.png)



### ARMAX(ARMA with eXogenous)

\begin{align*}
\text{Equation of ARMA} && Y_t &= \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} +
\theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} + \epsilon_t \\
\text{Equation of ARMAX} && Y_t &= \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} +
\theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} + \epsilon_t + \beta X \\
\end{align*}

### 실습: 통화량을 고려한 소비자지출 ARMAX 모델링


```python
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터 로딩 및 확인
source_url = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
raw = pd.read_stata(BytesIO(source_url))
raw.index = raw.time
raw_using = raw.loc['1960':'1981',["consump", "m2"]]
raw_using.plot()
plt.show()

# 모델링
## 회귀분석
fit = sm.OLS(raw_using.consump, sm.add_constant(raw_using.m2)).fit()
display(fit.summary())

## 잔차 확인
fit.resid.plot()
plt.show()

## 잔차 ACF/PACF
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(fit.resid, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual ACF")

sm.graphics.tsa.plot_pacf(fit.resid, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual PACF")
plt.tight_layout()
plt.show()

# 모델링
## ARIMAX
fit = sm.tsa.ARMA(raw_using.consump, (1,1), exog=raw_using.m2).fit()
display(fit.summary())

## 잔차 확인
fit.resid.plot()
plt.show()

## 잔차 ACF/PACF
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(fit.resid, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual ACF")

sm.graphics.tsa.plot_pacf(fit.resid, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual PACF")
plt.tight_layout()
plt.show()
```



![output_63_0](/assets/output_63_0.png)




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>consump</td>     <th>  R-squared:         </th> <td>   0.995</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.995</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.721e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 31 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>7.72e-101</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:46:10</td>     <th>  Log-Likelihood:    </th> <td> -434.48</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    88</td>      <th>  AIC:               </th> <td>   873.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    86</td>      <th>  BIC:               </th> <td>   877.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>  -61.7547</td> <td>    7.788</td> <td>   -7.930</td> <td> 0.000</td> <td>  -77.237</td> <td>  -46.273</td>
</tr>
<tr>
  <th>m2</th>    <td>    1.1406</td> <td>    0.009</td> <td>  131.182</td> <td> 0.000</td> <td>    1.123</td> <td>    1.158</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.847</td> <th>  Durbin-Watson:     </th> <td>   0.094</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.655</td> <th>  Jarque-Bera (JB):  </th> <td>   0.669</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.213</td> <th>  Prob(JB):          </th> <td>   0.716</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.984</td> <th>  Cond. No.          </th> <td>1.92e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.92e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




![output_63_2](/assets/output_63_2.png)





![output_63_3](/assets/output_63_3.png)




<table class="simpletable">
<caption>ARMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>consump</td>     <th>  No. Observations:  </th>    <td>88</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARMA(1, 1)</td>    <th>  Log Likelihood     </th> <td>-327.699</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>   <td>9.873</td>
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 31 Jul 2020</td> <th>  AIC                </th>  <td>665.398</td>
</tr>
<tr>
  <th>Time:</th>              <td>22:46:11</td>     <th>  BIC                </th>  <td>677.784</td>
</tr>
<tr>
  <th>Sample:</th>           <td>01-01-1960</td>    <th>  HQIC               </th>  <td>670.388</td>
</tr>
<tr>
  <th></th>                 <td>- 10-01-1981</td>   <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td>  -37.6135</td> <td>   36.043</td> <td>   -1.044</td> <td> 0.297</td> <td> -108.256</td> <td>   33.029</td>
</tr>
<tr>
  <th>m2</th>            <td>    1.1232</td> <td>    0.034</td> <td>   33.282</td> <td> 0.000</td> <td>    1.057</td> <td>    1.189</td>
</tr>
<tr>
  <th>ar.L1.consump</th> <td>    0.9330</td> <td>    0.043</td> <td>   21.867</td> <td> 0.000</td> <td>    0.849</td> <td>    1.017</td>
</tr>
<tr>
  <th>ma.L1.consump</th> <td>    0.3106</td> <td>    0.116</td> <td>    2.682</td> <td> 0.007</td> <td>    0.084</td> <td>    0.538</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>           1.0718</td> <td>          +0.0000j</td> <td>           1.0718</td> <td>           0.0000</td>
</tr>
<tr>
  <th>MA.1</th> <td>          -3.2199</td> <td>          +0.0000j</td> <td>           3.2199</td> <td>           0.5000</td>
</tr>
</table>




![output_63_5](/assets/output_63_5.png)





![output_63_6](/assets/output_63_6.png)




```python
# 좀 더 적합성능을 높이는 방법은? ARMA의 한계는 추세차분!

# SARIMAX 모델링
fit = sm.tsa.SARIMAX(raw_using.consump, exog=raw_using.m2, order=(1,0,0), seasonal_order=(1,0,1,4)).fit()
display(fit.summary())

# 잔차 확인
fit.resid.plot()
plt.show()

# 잔차 ACF/PACF
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(fit.resid, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual ACF")

sm.graphics.tsa.plot_pacf(fit.resid, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Residual PACF")
plt.tight_layout()
plt.show()
```


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>               <td>consump</td>             <th>  No. Observations:  </th>    <td>88</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 0, 0)x(1, 0, [1], 4)</td> <th>  Log Likelihood     </th> <td>-333.579</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Wed, 30 Sep 2020</td>         <th>  AIC                </th>  <td>677.158</td>
</tr>
<tr>
  <th>Time:</th>                       <td>00:27:58</td>             <th>  BIC                </th>  <td>689.545</td>
</tr>
<tr>
  <th>Sample:</th>                    <td>01-01-1960</td>            <th>  HQIC               </th>  <td>682.148</td>
</tr>
<tr>
  <th></th>                          <td>- 10-01-1981</td>           <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>               <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>m2</th>      <td>    0.3645</td> <td>    0.140</td> <td>    2.605</td> <td> 0.009</td> <td>    0.090</td> <td>    0.639</td>
</tr>
<tr>
  <th>ar.L1</th>   <td>    0.9987</td> <td>    0.004</td> <td>  270.804</td> <td> 0.000</td> <td>    0.991</td> <td>    1.006</td>
</tr>
<tr>
  <th>ar.S.L4</th> <td>    0.9541</td> <td>    0.060</td> <td>   15.855</td> <td> 0.000</td> <td>    0.836</td> <td>    1.072</td>
</tr>
<tr>
  <th>ma.S.L4</th> <td>   -0.4736</td> <td>    0.159</td> <td>   -2.985</td> <td> 0.003</td> <td>   -0.785</td> <td>   -0.163</td>
</tr>
<tr>
  <th>sigma2</th>  <td>   95.1782</td> <td>    9.448</td> <td>   10.074</td> <td> 0.000</td> <td>   76.661</td> <td>  113.696</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>47.49</td> <th>  Jarque-Bera (JB):  </th> <td>229.53</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.19</td>  <th>  Prob(JB):          </th>  <td>0.00</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>23.47</td> <th>  Skew:              </th>  <td>-0.99</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>10.66</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_64_1](/assets/output_64_1.png)





![output_64_2](/assets/output_64_2.png)



## 적분 선형확률과정(Integrated Linear Process)

### ARIMA(Auto-Regressive Integrated Moving Average)

> **"$ARIMA(p,d,q)$: 1이상의 차분이 적용된 $\Delta^d Y_t = (1-L)^d Y_{t}$가 알고리즘의 차수($p~and~q$)가 유한한 $AR(p)$와 $MA(q)$의 선형조합"**  
> - 비정상성인 시계열 데이터 $Y_t$를 차분한 결과로 만들어진 $\Delta Y_t = Y_t - Y_{t-1} = (1-L) Y_{t}$가 정상성인 데이터이고 ARMA 모형을 따르면 원래의 $Y_t$를 **ARIMA 모형**이라고 함  
**=> $d \geq 1$:** $Y_t$는 비정상성 시계열 데이터이다(단위근을 갖는다)  
> - $d$번 차분한 후 시계열 $\Delta^d Y_t$가 정상성인 데이터이고 ARMA(p,q) 모형을 따른다면 적분차수(Order of Integrarion)가 $d$인 **ARIMA(p,d,q)**로 표기함  
    - $p=0$: ARIMA(0,d,q) = IMA(d,q)
    - $q=0$: ARIMA(p,d,0) = ARI(p,d)  

| Parameters | Description |
|-----|----------------------------------|
| $p$ | order of the autoregressive part |
| $d$ | degree of differencing involved |
| $q$ | order of the moving average part |


- **ARIMA(0,1,1) = IMA(1,1)**
> **"자기상관계수(ACF)가 빠르게 감소하지 않는 것이 ARIMA와 같은 적분과정(Integrated Process)의 특징"**
> - 차분을 해야 정상성이 되는 $Y_t$이기에, 시차(Lag)가 증가해도 ACF가 1에 가까운 상관성을 유지하려 하기에 쉽게 감소하지 않음

\begin{align*}
Y_t &= Y_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1} \\
Y_t &= \epsilon_t+(1+\theta)\epsilon_{t-1}+(1+\theta)\epsilon_{t-2}+(1+\theta)\epsilon_{t-3}+\cdots \\
Corr(Y_t, Y_{t-1}) &= \rho_i \approx 1
\end{align*}


- **ARIMA(0,2,1) = IMA(2,1)**

\begin{align*}
\Delta^2 Y_t = (1-L)^2 Y_{t} = \epsilon_t + \theta_1 \epsilon_{t-1}\\
\end{align*}

### ARIMA 모형 차수결정 정리

> **1) 추정 및 예측을 하기 전에 파라미터에 따라 모형이 어떠한 결과를 도출할지 이해(예상) 필요**  
> **2) 결과이해(예상)는 기계의 실수를 방지하고 결과의 확신을 증가시킴**  

- **$p$, $q$ 파라미터 추론(by ACF and PACF):**  
1) 정상성 형태 변환: 차분/로그변환/계절성제거 등을 통해 데이터를 정상성 형태로 변환  
2) $ACF$, $PACF$를 도식화 하여 ARMA의 파라미터 차수를 추론  

<center><img src='Image/ARIMA_Pattern.png' width='500'></center>

- **$c$, $d$ 파라미터 이해: X가 반영되지 않고 추정된 시계열 알고리즘은 결국 상수항의 적합성을 높이는 것!**

> **"상수항(Const)인 $c$는 이론수식 복잡성으로 생략되기도 하나 존재가능"**  
> **"높은 차수의 차분($d$)은 예측 구간추정 범위를 급격하게 상승시킴"**   
> - $c = 0, d = 0$: 점추정은 0, 예측의 구간추정은 과거데이터의 표준편차
> - $c \neq 0, d = 0$: 점추정은 과거데이터의 평균, 예측의 구간추정은 과거데이터의 표준편차
> - $p \geq 2$: 특정 변동(계절성, 싸이클)을 반영한 예측을 위해선 2이상의 차수 필수
    - 예시: AR(2) 모형에서 $\phi_1^2+4\phi_2<0$를 만족해야 싸이클 형태로 예측되며 이때의 발생 싸이클은 다음과 같다
    \begin{align*}
    \frac{2\pi}{\text{arc cos}(-\phi_1(1-\phi_2)/(4\phi_2))}
    \end{align*}

| Parameters | Long-term Forecasts |
|-------------------|---------------------------------|
| $c = 0, \\ d = 0$ | go to zero |
| $c = 0, \\ d = 1$ | go to a non-zero constant |
| $c = 0, \\ d = 2$ | follow a straight line |
| $c \neq 0, \\ d = 0$ | go to the mean of the data |
| $c \neq 0, \\ d = 1$ | follow a straight line |
| $c \neq 0, \\ d = 2$ | follow a quadratic trend |


```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 2차누적/1차누적/미누적 데이터생성 및 적분차수 이해
np.random.seed(123)
y2 = sm.tsa.ArmaProcess([1], [1, 0.6]).generate_sample(100).cumsum().cumsum()
y1 = np.diff(y2)
y0 = np.diff(y1)

plt.figure(figsize=(10,8))
plt.subplot(311)
plt.title("ARIMA Simulation")
plt.plot(y2, 'o-')
plt.subplot(312)
plt.plot(y1, 'o-')
plt.subplot(313)
plt.plot(y0, 'o-')
plt.tight_layout()
plt.grid()
plt.show()
```



![output_68_0](/assets/output_68_0.png)




```python
# 2차누적/1차누적/미누적 데이터시각화
plt.figure(figsize=(10,8))
sm.tsa.graphics.plot_acf(y2, ax=plt.subplot(311))
plt.grid()
sm.tsa.graphics.plot_acf(y1, ax=plt.subplot(312))
plt.grid()
sm.tsa.graphics.plot_acf(y0, ax=plt.subplot(313))
plt.grid()
plt.tight_layout()
plt.show()
```



![output_69_0](/assets/output_69_0.png)



- **과차분(Over-differencing):**
> **"필요 적분차수 이상의 차분은 MA모형을 생성!"**
> - ARIMA(0,d,0) 모형을 따르는 $Y_t$를 $d$번 차분하면 백색잡음만 남음
> - 추가 1차분: MA(1), 추가 2차분: MA(2)
> - 과적합은 ACF/PACF의 수치를 오히려 증가시킬 수 있음

\begin{align*}
\Delta^d Y_t &= \epsilon_t \\
\Delta^{d+1} Y_t &= \epsilon_t - \epsilon_{t-1} \\
\Delta^{d+2} Y_t &= \epsilon_t - 2\epsilon_{t-1} + \epsilon_{t-2} \\
\end{align*}


```python
# 과적차분 데이터 이해
y1_minus = np.diff(y0)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.plot(y1_minus, 'o-')
plt.title("Over-differencing 1 (Data)")
plt.grid()
sm.tsa.graphics.plot_acf(y1_minus, ax=plt.subplot(212))
plt.title("Over-differencing 1 (ACF)")
plt.grid()
plt.show()

y2_minus = np.diff(y1_minus)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.plot(y2_minus, 'o-')
plt.title("Over-differencing 2 (Data)")
plt.grid()
sm.tsa.graphics.plot_acf(y2_minus, ax=plt.subplot(212))
plt.title("Over-differencing 2 (ACF)")
plt.grid()
plt.show()
```



![output_71_0](/assets/output_71_0.png)





![output_71_1](/assets/output_71_1.png)



- **단위근 존재 의미:**
> **"단위근을 갖는다는게 왜 비정상성 시계열 데이터라는 거지? -> $d$ 차분을 해야 정상성이 되니까!"**
> - "$Y_t$가 ARIMA(p,1,q)를 따른다" $=$ "$\Delta Y_t = Y_t - Y_{t-1}$가 정상성이며 ARMA(p,q)를 따른다"
> - 단위근이 있다는 것은 추세가 존재한다는 의미 -> 차분으로 추세제거 가능

\begin{align*}
\text{Main Equation of ARIMA} && \Delta Y_t = \phi_1 \Delta Y_{t-1} + \phi_2 \Delta Y_{t-2} + \cdots + \phi_p \Delta Y_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q} + \epsilon_t
\end{align*}

\begin{align*}
\Delta Y_t - \phi_1 \Delta Y_{t-1} - \phi_2 \Delta Y_{t-2} - \cdots - \phi_p \Delta Y_{t-p} &= \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q} + \epsilon_t \\
Y_t-Y_{t-1} - \phi_1 (Y_{t-1}-Y_{t-2}) - \phi_2 (Y_{t-2}-Y_{t-3}) - \cdots - \phi_p (Y_{t-p}-Y_{t-p-1}) &= \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q} \\
Y_t - (\phi_1+1)Y_{t-1} - (\phi_2-\phi_1)Y_{t-2} - \cdots - (\phi_p-\phi_{p-1})Y_{t-p} + \phi_pY_{t-p-1} &= \epsilon_t+\theta_1\epsilon_{t-1}+\theta_2 \epsilon_{t-2}+\cdots+\theta_q \epsilon_{t-q} \\
1-(\phi_1+1)Y-(\phi_2-\phi_1)Y^2-\cdots-(\phi_p-\phi_{p-1})Y^p+\phi_pY^{p+1} &= 0 \\
(1-Y)(1-\phi_1Y-\phi_2Y^2-\cdots-\phi_pY^p) &= 0 \\
\end{align*}

\begin{align*}
\text{Solution of ARIMA} && Y = 1 \text{ and } \sum_{i=1}^p \phi_i = 1
\end{align*}

\begin{align*}
1-\phi_1Y-\phi_2Y^2-\cdots-\phi_pY^p = 0 \\
\end{align*}

\begin{align*}
\text{Solution of ARMA} && \sum_{i=1}^p \phi_i = 1 \\
\end{align*}


- **ARIMA 표현식 정리:**

\begin{align*}
\text{Main Equation of ARMA} && Y_t &= \phi_1Y_{t-1} + \phi_2Y_{t-2} + \cdots + \phi_pY_{t-p} +
\theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} + \epsilon_t \\
&& \phi(L)Y_t &= \theta(L)\epsilon_t \\
\text{Main Equation of ARIMA} && (1-L)^d Y_t &= \phi_1 (1-L)^d Y_{t-1} + \phi_2 (1-L)^d Y_{t-2} + \cdots + \phi_p (1-L)^d Y_{t-p} +
\theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + \cdots + \theta_q\epsilon_{t-q} + \epsilon_t \\
&& \phi(L)(1-L)^dY_t &= \theta(L)\epsilon_t \\
\end{align*}

\begin{equation}
  \begin{array}{c c c c}
    (1-\phi_1L - \cdots - \phi_p L^p) & (1-L)^d Y_{t} &= (1 + \theta_1 L + \cdots + \theta_q L^q)\epsilon_t\\
    {\uparrow} & {\uparrow} & {\uparrow}\\
    \text{AR($p$)} & \text{$d$ differences} & \text{MA($q$)}\\
  \end{array}
\end{equation}

| Time Series Algorithms | ARIMA Expressions |
|-------------------------|-------------------|
| White noise | ARIMA(0,0,0) |
| Random walk | ARIMA(0,1,0) |
| Autoregression($AR(p)$) | ARIMA($p$,0,0) |
| Moving average($MA(q)$) | ARIMA(0,0,$q$) |

\begin{align*}
\text{Main Equation of ARIMA} && (1-\phi_1L - \cdots - \phi_p L^p) (1-L)^d Y_{t} &= (1 + \theta_1 L + \cdots + \theta_q L^q)\epsilon_t\\
\text{Main Equation of ARIMAX} && (1-\phi_1L - \cdots - \phi_p L^p) (1-L)^d Y_{t} &= \sum^{k}_{i=1}\beta_{i}Z_{it} + (1 + \theta_1 L + \cdots + \theta_q L^q)\epsilon_t\\
\end{align*}



```python
import pandas as pd
import numpy as np
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test
```


```python
# ARIMA 모형의 한계
# 데이터 로딩 및 시간변수 반영
raw_set = datasets.get_rdataset("accdeaths", package="MASS")
raw = raw_set.data
raw.time = pd.date_range('1973-01-01', periods=len(raw), freq='M')
raw['month'] = raw.time.dt.month

# 데이터 확인
display(raw.tail())
plt.plot(raw.time, raw.value)
plt.show()

# 정상성 확인
display(stationarity_adf_test(raw.value, []))
display(stationarity_kpss_test(raw.value, []))
sm.graphics.tsa.plot_acf(raw.value, lags=50, use_vlines=True)
plt.tight_layout()
plt.show()
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
      <th>time</th>
      <th>value</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>1978-08-31</td>
      <td>9827</td>
      <td>8</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1978-09-30</td>
      <td>9110</td>
      <td>9</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1978-10-31</td>
      <td>9070</td>
      <td>10</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1978-11-30</td>
      <td>8633</td>
      <td>11</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1978-12-31</td>
      <td>9240</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




![output_74_1](/assets/output_74_1.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-2.54</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.11</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>59.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.55</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>870.44</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.28</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_74_4](/assets/output_74_4.png)



### SARIMA(Seasonal ARIMA)

> **"ARIMA 모형은 Non-seasonal 데이터 또는 Non-seasonal ARIMA 모델을 가정 -> 계절성 패턴 반영 모델 필요!"**  
- SARIMAX 클래스 이용하면 Multiplicated SARIMA(p,d,q)x(P,D,Q,m) 모형 추정 및 예측 가능  
- SARIMAX의 fit 메서드는 모수를 추정하여 그 결과를 SARIMAXResult 클래스 인스턴스로 반환

\begin{align*}
\text{SARIMA} && \underbrace{(p, d, q)} && \underbrace{(P, D, Q)_m} \\
&& {\uparrow} && {\uparrow} \\
&& \text{Non-seasonal part} && \text{Seasonal part} \\
&& \text{of the model} && \text{of the model} \\
\end{align*}

\begin{align*}
\text{where } m = \text{ seasonal lag of observations.}
\end{align*}

\begin{align*}
\text{Equation: ARIMA(p,d,q)} && (1-\phi_1L - \cdots - \phi_p L^p) (1-L)^d Y_{t} &=
  (1 + \theta_1 L + \cdots + \theta_q L^q) \epsilon_t\\
\text{Equation: ARIMA(1,1,1)} && (1 - \phi_{1}L) (1 - L)Y_{t} &= (1 + \theta_{1}L) \epsilon_{t}. \\
\text{Equation: SARIMA(p,d,q)(P,D,Q)}_m && (1-\phi_1L - \cdots - \phi_p L^p) (1 - \Phi_{1}L^{m} - \Phi_{2}L^{2m} - \cdots - \Phi_{P}L^{Pm}) (1-L)^d (1-L^{m})^D Y_{t} &=
  (1 + \theta_1 L + \cdots + \theta_q L^q) (1 + \Theta_{1}L^{m} + \Theta_{2}L^{2m} + \cdots + \Theta_{Q}L^{Qm}) \epsilon_t\\
\text{Equation: SARIMA(1,1,1)(1,1,1)}_4 && (1 - \phi_{1}L)~(1 - \Phi_{1}L^{4}) (1 - L) (1 - L^{4})Y_{t} &=
  (1 + \theta_{1}L)~ (1 + \Theta_{1}L^{4})\epsilon_{t}\\
\text{Equation: SARIMA(1,2,1)(1,2,1)}_4 && (1 - \phi_{1}L)~(1 - \Phi_{1}L^{4}) (1 - L)^2 (1 - L^{4})^2 Y_{t} &=
  (1 + \theta_{1}L)~ (1 + \Theta_{1}L^{4})\epsilon_{t}.
\end{align*}

#### Simple SARIMA: 계절성 시차에서만 ACF가 유의하지 않음  
- **SARIMA(0,0,0)(0,0,1,12):** 각 월의 시계열 자료의 값이 현재의 백색잡음과 작년 동월 백색잡음에 의해 생성
> - ACF 그래프에서 계절성시차(Lag12)에서의 계수가 유의수준을 벗어난 증가를 보임(다른 시차에서는 유의수준 내 존재)  
> - PACF 그래프에서 반복되는 계절성시차들의 지수적 감소를 보임   

\begin{align*}
\text{Main Equation} && Y_t &= (1 + \Theta_{1}L^{12})\epsilon_{t} \\
&& &= \epsilon_t + \Theta \epsilon_{t-12} \\
\text{Covariance} && Cov(Y_t, Y_{t-1}) &= \text{Cov}( \epsilon_t + \Theta \epsilon_{t-12} ,  \epsilon_{t-1} + \Theta \epsilon_{t-13} ) = 0 \\
&& Cov(Y_t, Y_{t-12}) &= \text{Cov}( \epsilon_t + \Theta \epsilon_{t-12} ,  \epsilon_{t-12} + \Theta \epsilon_{t-24} ) = -\Theta \sigma_e^2 \\
\text{Autocorrelation} && \rho_{k \cdot 12} &= \dfrac{\Theta_k + \Theta_{1}\Theta_{k+1} + \Theta_{2}\Theta_{k+2} + \cdots + \Theta_{Q-k}\Theta_{k+Q}}{1 + \Theta_1^2 +\Theta_2^2 + \cdots + \Theta_Q^2} \\
\end{align*}

- **SARIMA(0,0,0)(1,0,0,12):** 각 월의 시계열 자료의 값이 작년 동월 자료값과 현재의 백색잡음에 의해 생성
> - ACF 그래프에서 반복되는 계절성시차들의 지수적 감소를 보임  
> - PACF 그래프에서 계절성시차(Lag12)에서의 계수가 유의수준을 벗어난 증가를 보임(다른 시차에서는 유의수준 내 존재)  

\begin{align*}
\text{Main Equation} && (1 - \Phi_{1}L^{12}) Y_{t} &= \epsilon_t \\
&& Y_t &= \Phi Y_{t-12} + \epsilon_t \\
\text{Stationary Condition} && \Phi &< 1 \\
\text{Autocorrelation} && \rho_{k \cdot 12} &= (-1)^{k+1}\Phi^k \\
\end{align*}

- **SARIMA(0,0,0)(P,0,Q,12):**  

\begin{align*}
\text{Main Equation} && (1 - \Phi_{1}L^{12} - \Phi_{2}L^{24} - \cdots - \Phi_{P}L^{12P}) Y_{t} &= (1 + \Theta_{1}L^{12} + \Theta_{2}L^{24} + \cdots + \Theta_{Q}L^{12Q})\epsilon_{t} \\
&& Y_t - \Phi_1 Y_{t-12} - \Phi_2 Y_{t-24} - \cdots - \Phi_P Y_{t-12P} &= \epsilon_t + \Theta_1 \epsilon_{t-12} + \Theta_2 \epsilon_{t-24} + \cdots + \Theta_Q \epsilon_{t-12Q} \\
\end{align*}

- **SARIMA(0,0,0)(P,1,Q,12):**  

\begin{align*}
\text{Main Equation} && (1 - \Phi_{1}L^{12} - \Phi_{2}L^{24} - \cdots - \Phi_{P}L^{12P}) (1-L^{12}) Y_{t} &= (1 + \Theta_{1}L^{12} + \Theta_{2}L^{24} + \cdots + \Theta_{Q}L^{12Q})\epsilon_{t} \\
&& (Y_t-Y_{t-12}) - \Phi_1 (Y_{t-12}-Y_{t-24}) - \cdots - \Phi_P (Y_{t-12P}-Y_{t-12(P+1)}) \\
&&= \epsilon_t + \Theta_1 \epsilon_{t-12} + \Theta_2 \epsilon_{t-24} + \cdots + \Theta_Q \epsilon_{t-12Q} \\
\end{align*}

- **SARIMA(0,0,0)(0,1,1,12):**  

\begin{align*}
\text{Main Equation} && (1-L^{12}) Y_{t} &= (1 + \Theta_{1}L^{12})\epsilon_{t} \\
&& Y_t-Y_{t-12} &= \epsilon_t + \Theta_1 \epsilon_{t-12} \\
&& Y_t &= Y_{t-12} + \epsilon_t + \Theta_1 \epsilon_{t-12} \\
\end{align*}

- **SARIMA(0,0,0)(1,1,0,12):**  

\begin{align*}
\text{Main Equation} && (1 - \Phi_{1}L^{12}) (1-L^{12}) Y_{t} &= \epsilon_{t} \\
&& (Y_t-Y_{t-12}) - \Phi_1 (Y_{t-12}-Y_{t-24}) &= \epsilon_t \\
&& Y_t- (1 + \Phi_1)Y_{t-12} + \Phi_1 Y_{t-24} &= \epsilon_t \\
&& Y_t &= (1 + \Phi_1)Y_{t-12} - \Phi_1 Y_{t-24} + \epsilon_t \\
\end{align*}


- **계절성 차수 추정 정리:** 계절성 부분의 AR과 MA 차수는 ACF/PACF의 계절성 시차(Lag) 형태로 파악 가능  

| Parameters | Description |
|-------|-------------------------------------------------------|
| $p$ | Trend autoregression order |
| $d$ | Trend difference order |
| $q$ | Trend moving average order |
| $m$ | the number of time steps for a single seasonal period |
| $P$ | Seasonal autoregression order |
| $D$ | Seasonal difference order |
| $Q$ | Seasonal moving average order |

> **예시1:** SARIMA(0,0,0)(0,0,1$)_{12}$  
> - ACF 그래프에서 계절성시차(Lag12)에서의 계수가 유의수준을 벗어난 증가를 보임(다른 시차에서는 유의수준 내 존재)  
> - PACF 그래프에서 반복되는 계절성시차들의 지수적 감소를 보임  

> **예시2:** SARIMA(0,0,0)(1,0,0$)_{12}$  
> - ACF 그래프에서 반복되는 계절성시차들의 지수적 감소를 보임  
> - PACF 그래프에서 계절성시차(Lag12)에서의 계수가 유의수준을 벗어난 증가를 보임(다른 시차에서는 유의수준 내 존재)  



```python
import pandas as pd
import numpy as np
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
```


```python
# SARIMA(0,0,0)(0,0,1,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([])
ma_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_78_0](/assets/output_78_0.png)



    <ipython-input-2-0a11545eb78d>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-2-0a11545eb78d>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))




![output_78_2](/assets/output_78_2.png)




```python
# SARIMA(0,0,0)(1,0,0,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_79_0](/assets/output_79_0.png)



    <ipython-input-3-426d870da023>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-3-426d870da023>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))




![output_79_2](/assets/output_79_2.png)




```python
# SARIMA(0,0,0)(0,1,1,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99])
ma_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_80_0](/assets/output_80_0.png)



    <ipython-input-4-5d93f5fcaac3>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-4-5d93f5fcaac3>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\regression\linear_model.py:1406: RuntimeWarning: invalid value encountered in sqrt
      return rho, np.sqrt(sigmasq)




![output_80_2](/assets/output_80_2.png)




```python
# SARIMA(0,0,0)(1,1,0,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
ar_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.95])
ma_params = np.array([])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_81_0](/assets/output_81_0.png)



    <ipython-input-5-ae688816198f>:14: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-5-ae688816198f>:20: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\regression\linear_model.py:1406: RuntimeWarning: invalid value encountered in sqrt
      return rho, np.sqrt(sigmasq)




![output_81_2](/assets/output_81_2.png)



#### Multiplicated SARIMA: 계절성 시차와 그 주변의 시차에서도 ACF가 유의하지 않음  
- **SARIMA(0,0,1)(0,0,1,12):** 계절주기 12고 계절주기 자료들 간에는 ARIMA(0,0,1), 비계절 자료들 간에는 ARIMA(0,0,1)  
> - 시차(Lag)가 1, 11, 12, 13인 경우를 제외하고는 자기상관계수가 모두 0  

\begin{align*}
\text{Main Equation} && Y_t &= (1+\theta L)(1+\Theta L^{12}) \epsilon_t = \epsilon_t + \theta \epsilon_{t-1} + \Theta \epsilon_{t-12} + \theta\Theta \epsilon_{t-13} \\
\text{Autocorrelation} && \rho_1 &= -\dfrac{\theta}{1+\theta^2} \\
&& \rho_{11} = \rho_{13} &= \dfrac{\theta\Theta}{(1+\theta^2)(1+\Theta^2)} \\
&& \rho_{12} &= -\dfrac{\Theta}{1+\Theta^2} \\
\end{align*}

- **SARIMA(0,0,1)(1,0,0,12):** 계절주기 12고 계절주기 자료들 간에는 ARIMA(1,0,0), 비계절 자료들 간에는 ARIMA(0,0,1)  
> - 시차(Lag)가 12의 배수와 그 앞/뒤인 경우(12$k$, 12$k$+1, 12$k$-1)를 제외하고는 자기상관계수가 모두 0  

\begin{align*}
\text{Main Equation} && (1 - \Phi L^{12})Y_t &= (1 + \theta L) \epsilon_t \\
&& Y_t &=  \Phi Y_{t-12} + \epsilon_t + \theta \epsilon_{t-1} \\
\text{Autocorrelation} && \rho_{12k} &= (-1)^{k+1}\Phi^k \\
&& \rho_1 &= -\dfrac{\theta}{1+\theta^2} \\
&& \rho_{12k-1} = \rho_{12k+1} &= (-1)^{k+1}\dfrac{\theta}{1+\theta^2} \Phi^k \\
\end{align*}

- **SARIMA(0,1,0)(0,1,0,12):** 계절주기 12고 계절주기 간 1차 차분 ARIMA(0,1,0), 비계절 자료들 간 1차 차분 ARIMA(0,1,0)  
> - 시계열을 1차 차분하고 그 시계열을 다시 12간격 차분하면 백색잡음    

\begin{align*}
\text{Main Equation} && (1 - L^{12})(1 - L)Y_t &= \epsilon_t \\
&& (1 - L^{12})(Y_t - Y_{t-1}) &= \epsilon_t \\
&& Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13} &= \epsilon_t \\
&& Y_t &= Y_{t-1} + Y_{t-12} - Y_{t-13} + \epsilon_t \\
\end{align*}

- **SARIMA(0,1,1)(0,1,1,12):** 계절주기 12고 계절/비계절 자료 모두 1차 차분 ARIMA(0,1,1)  
> - 시계열을 1차 차분하고 그 시계열을 다시 12간격 차분하면 계절/비계절 자료 모두 ARIMA(0,0,1)    

\begin{align*}
\text{Main Equation} && (1 - L^{12})(1 - L)Y_t &= (1 + \theta L)(1 + \Theta L^{12}) \epsilon_t \\
&& (1 - L^{12})(Y_t - Y_{t-1}) &= \epsilon_t + \theta \epsilon_{t-1} + \Theta \epsilon_{t-12} + \theta\Theta \epsilon_{t-13} \\
&& Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13} &= \epsilon_t + \theta \epsilon_{t-1} + \Theta \epsilon_{t-12} + \theta\Theta \epsilon_{t-13} \\
&& Y_t &= Y_{t-1} + Y_{t-12} - Y_{t-13} + \epsilon_t + \theta \epsilon_{t-1} + \Theta \epsilon_{t-12} + \theta\Theta \epsilon_{t-13} \\
\end{align*}

- **SARIMA(1,1,0)(1,1,0,12):** 계절주기 12고 계절/비계절 자료 모두 1차 차분 ARIMA(1,1,0)  
> - 시계열을 1차 차분하고 그 시계열을 다시 12간격 차분하면 계절/비계절 자료 모두 ARIMA(1,0,0)    

\begin{align*}
\text{Main Equation} && (1 - \phi L)(1 - \Phi L^{12})(1 - L^{12})(1 - L)Y_t &= \epsilon_t \\
&& (1 - \phi L)(1 - \Phi L^{12})(1 - L^{12})(Y_t - Y_{t-1}) &= \epsilon_t \\
&& (1 - \phi L)(1 - \Phi L^{12})(Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}) &= \epsilon_t \\
&& (1 - \phi L)(Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13} - \Phi Y_{t-12} + \Phi Y_{t-13} + \Phi Y_{t-24} - \Phi Y_{t-25})  Y_t &= \epsilon_t \\
&& (1 - \phi L)(Y_t - Y_{t-1} - (1 + \Phi)Y_{t-12} + (1 + \Phi)Y_{t-13} + \Phi Y_{t-24} - \Phi Y_{t-25})  Y_t &= \epsilon_t \\
&& Y_t - Y_{t-1} - (1 + \Phi)Y_{t-12} + (1 + \Phi)Y_{t-13} + \Phi Y_{t-24} - \Phi Y_{t-25} &\\
&& - \phi Y_{t-1} +\phi Y_{t-2} +\phi (1 + \Phi)Y_{t-13} -\phi (1 + \Phi)Y_{t-14} -\phi \Phi Y_{t-25} +\phi \Phi Y_{t-26} &= \epsilon_t \\
&& Y_t - (1+\phi) Y_{t-1} +\phi Y_{t-2} - (1 + \Phi)Y_{t-12} +((1 + \Phi)+\phi (1 + \Phi))Y_{t-13} &\\
&& -\phi (1 + \Phi)Y_{t-14} + \Phi Y_{t-24} - (\Phi+\phi \Phi) Y_{t-25} +\phi \Phi Y_{t-26} &= \epsilon_t \\
\end{align*}


```python
# SARIMA(0,0,1)(0,0,1,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
phi, Phi = 0, 0
theta, Theta = 0.5, 0.8
ar_params = np.array([])
ma_params = np.array([theta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Theta, theta*Theta])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_83_0](/assets/output_83_0.png)



    <ipython-input-3-d5b2fcaf7280>:16: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-3-d5b2fcaf7280>:22: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))




![output_83_2](/assets/output_83_2.png)




```python
# SARIMA(0,0,1)(1,0,0,12)
plt.figure(figsize=(10, 4))
np.random.seed(123)
phi, Phi = 0, 0.75
theta, Theta = 0.5, 0
ar_params = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Phi])
ma_params = np.array([theta])
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]
y = sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 9))
plt.subplot(411)
plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical ACF of an SARIMA process")

plt.subplot(412)
plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical PACF of an SARIMA process")

sm.graphics.tsa.plot_acf(y, lags=50, ax=plt.subplot(413))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling ACF of an SARIMA process")

sm.graphics.tsa.plot_pacf(y, lags=50, ax=plt.subplot(414))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("Sampling PACF of an SARIMA process")
plt.tight_layout()
plt.show()
```



![output_84_0](/assets/output_84_0.png)



    <ipython-input-4-c256aca65ca6>:16: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).acf(lags=50))
    <ipython-input-4-c256aca65ca6>:22: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      plt.stem(sm.tsa.ArmaProcess(ar, ma).pacf(lags=50))




![output_84_2](/assets/output_84_2.png)



### SARIMA 모델링 정리

- **예시:**  

> (원 데이터)  
> <center><img src='Image/TS_SARIMA_Example1.png' width='600'></center>  

> - $p:2?$ (PACF 기준 lag 2까지 유의하고 그 뒤로는 유의하지 않음)  
> - $d:1?$ (ADF가 대중가설을 기각하므로 추세 1차 차분)  
> - $q:1?$ (ACF 기준 필요성 인지)  
> - $P:1?$ (PACF 기준 lag 24 간격 유의성으로 필요성 인지)    
> - $D:1?$ (계절성 차분 필요함 인지)  
> - $Q:2?$ (ACF 기준 lag 24 간격 유의성으로 필요성 인지)  
> - $m:24?$ (ACF/PACF 기준 lag 24 간격으로 유의한 진동 존재)  

> (계절차분 후 데이터)  
> <center><img src='Image/TS_SARIMA_Example2.png' width='600'></center>  

> - $p:1?$ (PACF 기준 lag 1까지 유의하고 그 뒤로는 유의하지 않음)  
> - $d:1?$ (ADF가 대중가설을 기각하고 그래프 상 추세가 보이므로 일단 추세 1차 차분)  
> - $q:3?$ (ACF 기준 필요성 인지)  
> - $P:2?$ (PACF 기준 lag 24 간격 유의성으로 필요성 인지)    
> - $D:1$ (계절성 차분 필요)  
> - $Q:0?$ (ACF 기준 lag 24 간격 유의성으로 필요성 인지)  
> - $m:24$ (ACF/PACF 기준 lag 24 간격으로 유의한 진동 크게 사라짐)  

> (계절성 및 추세차분 후 데이터)  
> <center><img src='Image/TS_SARIMA_Example3.png' width='600'></center>  

> - $p:max4$ (PACF 기준 lag 4까지 유의하고 그 뒤로는 유의하지 않음)  
> - $d:1$ (ADF가 대중가설을 기각하고 그래프 상 추세도 없어졌으므로 추세 1차 차분 확정)  
> - $q:max4$ (ACF 기준 lag 4까지 유의하고 그 뒤로는 유의하지 않음)  
> - $P:max2$ (PACF 기준 lag 24 간격으로 2번정도 유의함)    
> - $D:1$ (계절성 차분 필요함 인지)  
> - $Q:max1$ (ACF 기준 lag 24 간격으로 1번정도 유의함)  
> - $m:24$ (lag 24 간격으로 진동 존재)  

> (잔차검증)
> <center><img src='Image/TS_SARIMA_Example4.png' width='600'></center>  

### 실습: 호흡기질환 사망자수 SARIMA 모델링


```python
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터로딩 및 확인
data = sm.datasets.get_rdataset("deaths", "MASS")
raw = data.data
raw.value = np.log(raw.value)
raw.plot(x='time', y='value')
plt.show()

# ACF/PACF 확인
plt.figure(figsize=(10, 8))
sm.graphics.tsa.plot_acf(raw.value.values, lags=50, ax=plt.subplot(211))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("ACF")

sm.graphics.tsa.plot_pacf(raw.value.values, lags=50, ax=plt.subplot(212))
plt.xlim(-1, 51)
plt.ylim(-1.1, 1.1)
plt.title("PACF")
plt.tight_layout()
plt.show()

# ARMA(1,1) 모델링
fit = sm.tsa.SARIMAX(raw.value, trend='c', order=(1,0,1), seasonal_order=(0,0,0,0)).fit()
display(fit.summary())

# 잔차진단
fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()
```



![output_87_0](/assets/output_87_0.png)





![output_87_1](/assets/output_87_1.png)




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>value</td>      <th>  No. Observations:  </th>   <td>72</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 0, 1)</td> <th>  Log Likelihood     </th> <td>31.982</td>
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  AIC                </th> <td>-55.965</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:18:33</td>     <th>  BIC                </th> <td>-46.858</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>-52.339</td>
</tr>
<tr>
  <th></th>                       <td> - 72</td>      <th>                     </th>    <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    2.4194</td> <td>    0.765</td> <td>    3.161</td> <td> 0.002</td> <td>    0.919</td> <td>    3.920</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>    0.6818</td> <td>    0.100</td> <td>    6.846</td> <td> 0.000</td> <td>    0.487</td> <td>    0.877</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>    0.4668</td> <td>    0.116</td> <td>    4.027</td> <td> 0.000</td> <td>    0.240</td> <td>    0.694</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0236</td> <td>    0.004</td> <td>    5.416</td> <td> 0.000</td> <td>    0.015</td> <td>    0.032</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>161.08</td> <th>  Jarque-Bera (JB):  </th> <td>12.13</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>0.00</td>  <th>  Prob(JB):          </th> <td>0.00</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th>  <td>1.00</td>  <th>  Skew:              </th> <td>0.94</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.99</td>  <th>  Kurtosis:          </th> <td>3.73</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_87_3](/assets/output_87_3.png)




```python
# SARIMA 모델링
fit = sm.tsa.SARIMAX(raw.value, trend='c', order=(1,0,1), seasonal_order=(1,1,1,12)).fit()
display(fit.summary())

# 잔차진단
fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()
```


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>                <td>value</td>             <th>  No. Observations:  </th>   <td>72</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 0, 1)x(1, 1, 1, 12)</td> <th>  Log Likelihood     </th> <td>52.592</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Thu, 01 Oct 2020</td>        <th>  AIC                </th> <td>-93.183</td>
</tr>
<tr>
  <th>Time:</th>                       <td>11:22:23</td>            <th>  BIC                </th> <td>-80.617</td>
</tr>
<tr>
  <th>Sample:</th>                         <td>0</td>               <th>  HQIC               </th> <td>-88.268</td>
</tr>
<tr>
  <th></th>                              <td> - 72</td>             <th>                     </th>    <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>    <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -0.0481</td> <td>    0.022</td> <td>   -2.216</td> <td> 0.027</td> <td>   -0.091</td> <td>   -0.006</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>   -0.1566</td> <td>    0.435</td> <td>   -0.360</td> <td> 0.719</td> <td>   -1.009</td> <td>    0.696</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>    0.5725</td> <td>    0.272</td> <td>    2.105</td> <td> 0.035</td> <td>    0.039</td> <td>    1.106</td>
</tr>
<tr>
  <th>ar.S.L12</th>  <td>   -0.2875</td> <td>    0.203</td> <td>   -1.414</td> <td> 0.157</td> <td>   -0.686</td> <td>    0.111</td>
</tr>
<tr>
  <th>ma.S.L12</th>  <td>   -0.9648</td> <td>    4.037</td> <td>   -0.239</td> <td> 0.811</td> <td>   -8.878</td> <td>    6.948</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0066</td> <td>    0.026</td> <td>    0.256</td> <td> 0.798</td> <td>   -0.044</td> <td>    0.057</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>28.58</td> <th>  Jarque-Bera (JB):  </th> <td>74.59</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.91</td>  <th>  Prob(JB):          </th> <td>0.00</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.64</td>  <th>  Skew:              </th> <td>0.99</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.32</td>  <th>  Kurtosis:          </th> <td>8.09</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_88_1](/assets/output_88_1.png)



### 실습: 항공사 승객수요 SARIMA 모델링


```python
# 라이브러리 호출
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터 준비
data = sm.datasets.get_rdataset("AirPassengers")
raw = data.data.copy()

# 데이터 전처리
## 시간 인덱싱
if 'time' in raw.columns:
    raw.index = pd.date_range(start='1/1/1949', periods=len(raw['time']), freq='M')
    del raw['time']

## 정상성 확보
plt.figure(figsize=(12,8))
raw.plot(ax=plt.subplot(221), title='Y', legend=False)
np.log(raw).plot(ax=plt.subplot(222), title='log(Y)', legend=False)
raw.diff(1).plot(ax=plt.subplot(223), title='diff1(Y)', legend=False)
np.log(raw).diff(1).plot(ax=plt.subplot(224), title='diff1(log(Y))', legend=False)
plt.show()
```



![output_90_0](/assets/output_90_0.png)




```python
# ARIMA 모델링 (raw)
fit = sm.tsa.SARIMAX(raw.value, trend='c', order=(1,1,1), seasonal_order=(0,0,0,0)).fit()
display(fit.summary())

# 잔차진단
fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()

# ARIMA 모델링 (log(raw))
fit = sm.tsa.SARIMAX(np.log(raw.value), trend='c', order=(1,1,1), seasonal_order=(0,0,0,0)).fit()
display(fit.summary())

# 잔차진단
fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()
```


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>value</td>      <th>  No. Observations:  </th>    <td>144</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 1)</td> <th>  Log Likelihood     </th> <td>-694.060</td>
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  AIC                </th> <td>1396.121</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:27:12</td>     <th>  BIC                </th> <td>1407.972</td>
</tr>
<tr>
  <th>Sample:</th>             <td>01-31-1949</td>    <th>  HQIC               </th> <td>1400.937</td>
</tr>
<tr>
  <th></th>                   <td>- 12-31-1960</td>   <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    3.6187</td> <td>    5.001</td> <td>    0.724</td> <td> 0.469</td> <td>   -6.183</td> <td>   13.421</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>   -0.4768</td> <td>    0.128</td> <td>   -3.736</td> <td> 0.000</td> <td>   -0.727</td> <td>   -0.227</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>    0.8646</td> <td>    0.080</td> <td>   10.746</td> <td> 0.000</td> <td>    0.707</td> <td>    1.022</td>
</tr>
<tr>
  <th>sigma2</th>    <td>  958.4116</td> <td>  107.040</td> <td>    8.954</td> <td> 0.000</td> <td>  748.616</td> <td> 1168.207</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>318.10</td> <th>  Jarque-Bera (JB):  </th> <td>2.17</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>0.00</td>  <th>  Prob(JB):          </th> <td>0.34</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th>  <td>7.01</td>  <th>  Skew:              </th> <td>-0.21</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.00</td>  <th>  Kurtosis:          </th> <td>3.43</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_91_1](/assets/output_91_1.png)




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>value</td>      <th>  No. Observations:  </th>    <td>144</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 1)</td> <th>  Log Likelihood     </th>  <td>124.804</td>
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 01 Oct 2020</td> <th>  AIC                </th> <td>-241.608</td>
</tr>
<tr>
  <th>Time:</th>                <td>11:27:12</td>     <th>  BIC                </th> <td>-229.756</td>
</tr>
<tr>
  <th>Sample:</th>             <td>01-31-1949</td>    <th>  HQIC               </th> <td>-236.792</td>
</tr>
<tr>
  <th></th>                   <td>- 12-31-1960</td>   <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.0155</td> <td>    0.016</td> <td>    0.963</td> <td> 0.336</td> <td>   -0.016</td> <td>    0.047</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>   -0.5826</td> <td>    0.170</td> <td>   -3.430</td> <td> 0.001</td> <td>   -0.915</td> <td>   -0.250</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>    0.8502</td> <td>    0.101</td> <td>    8.455</td> <td> 0.000</td> <td>    0.653</td> <td>    1.047</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0102</td> <td>    0.002</td> <td>    5.871</td> <td> 0.000</td> <td>    0.007</td> <td>    0.014</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>355.81</td> <th>  Jarque-Bera (JB):  </th> <td>5.93</td>
</tr>
<tr>
  <th>Prob(Q):</th>                 <td>0.00</td>  <th>  Prob(JB):          </th> <td>0.05</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th>  <td>1.07</td>  <th>  Skew:              </th> <td>0.04</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>     <td>0.81</td>  <th>  Kurtosis:          </th> <td>2.01</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_91_3](/assets/output_91_3.png)




```python
# SARIMA 모델링 (log(raw))
fit = sm.tsa.SARIMAX(np.log(raw.value), trend='c', order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
display(fit.summary())

# 잔차진단
fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()
```


<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>                <td>value</td>             <th>  No. Observations:  </th>    <td>144</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 1)x(1, 1, 1, 12)</td> <th>  Log Likelihood     </th>  <td>245.169</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Thu, 01 Oct 2020</td>        <th>  AIC                </th> <td>-478.338</td>
</tr>
<tr>
  <th>Time:</th>                       <td>11:30:21</td>            <th>  BIC                </th> <td>-461.087</td>
</tr>
<tr>
  <th>Sample:</th>                    <td>01-31-1949</td>           <th>  HQIC               </th> <td>-471.329</td>
</tr>
<tr>
  <th></th>                          <td>- 12-31-1960</td>          <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -0.0002</td> <td>    0.001</td> <td>   -0.209</td> <td> 0.835</td> <td>   -0.002</td> <td>    0.002</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>    0.1645</td> <td>    0.211</td> <td>    0.779</td> <td> 0.436</td> <td>   -0.249</td> <td>    0.578</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>   -0.5627</td> <td>    0.184</td> <td>   -3.064</td> <td> 0.002</td> <td>   -0.923</td> <td>   -0.203</td>
</tr>
<tr>
  <th>ar.S.L12</th>  <td>   -0.1094</td> <td>    0.201</td> <td>   -0.545</td> <td> 0.586</td> <td>   -0.503</td> <td>    0.284</td>
</tr>
<tr>
  <th>ma.S.L12</th>  <td>   -0.4911</td> <td>    0.218</td> <td>   -2.254</td> <td> 0.024</td> <td>   -0.918</td> <td>   -0.064</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0013</td> <td>    0.000</td> <td>    8.484</td> <td> 0.000</td> <td>    0.001</td> <td>    0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>37.22</td> <th>  Jarque-Bera (JB):  </th> <td>3.50</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.60</td>  <th>  Prob(JB):          </th> <td>0.17</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.61</td>  <th>  Skew:              </th> <td>-0.02</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.11</td>  <th>  Kurtosis:          </th> <td>3.80</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




![output_92_1](/assets/output_92_1.png)



## 선형확률과정의 분석싸이클

### 분석싸이클 정리(Non-seasonal)

**1. 분석싸이클 제시: [<박스-젠킨스 방법론>](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method)**

> **1) 모형의 모수추정(Model Identification):**
    - 시계열 데이터의 정상성을 확인하고 계절변동이 있는지도 확인  
    - ACF/PACF 를 사용해서 자기회귀이동평균 모형 p,q 차수를 결정   

> **2) 모델링 및 검증(Parameter Estimation):**  
    - 회귀분석과 기계학습 등의 검증지표를 마찬가지로 사용  
    - 모형 추정은 최소제곱방법과 유사하지만 가우스-뉴튼 아이디어에 기초한 수치해석방법을 적용  

> **3) 잔차진단(Model Diagnostics):**  
    - 자기회귀이동평균 모형을 적용시키고 남은 잔차의 정상성을 확인하는데 중점  
    - 잔차가 서로 독립이고 시간에 따라 평균과 분산이 일정한지 검증
    - 시계열 데이터의 자기상관을 검정하기 위해 다양한 검정통계량을 사용하는데,  
    Ljung-Box 통계량, 평균과 분산이 일정한지, ACF/PACF 사용하여 추가적으로 모형에 누락된 것이 없는지 검정


**2. 분석싸이클 일반화**

> **0) 데이터 전처리 및 시각화를 통해 Outlier 확인/변경/제거**

> **1) 비정상 과정에서 정상 과정 추출**
- 결정론적 추세나 확률적 추세가 있는지 확인
    - 결정론적 추세는 회귀분석, 다항식 등으로 모형화 후 이를 분리
    - 확률적 추세인 경우, 즉 ARIMA 모형인 경우에는 ADF(Augmented Dickey Fuller) 검정을 사용하여 적분차수(Order of Integration)을 알아내서 차분  

> **2) 정규성 확인**
- 정규성 검정을 통해 자료의 분포가 정규 분포인지 확인
    - 일반 선형 확률 과정인 경우에는 전체 시계열이 가우시안 백색 잡음의 선형 조합으로 이루어지기 때문에 시계열 자체도 가우시안 정규 분포
    - ARIMA 모형 등의 일반 선형 확률 과정으로 모형화하려면 우선 정규성 검정(Normality Test)을 사용하여 분포가 정규 분포인지 확인
- 만약 시계열 자료의 분포가 로그 변환이나 Box-Cox 변환을 사용하여 정규성이 개선된다면 이러한 변환을 사용 가능

> **3) 정상 과정에 대한 ARMA 모형 차수 결정**
- ACF/PACF 분석으로 AR(p) 모형 또는 MA(q) 모형 결정
    - ACF가 특정 차수 이상에서 없어지는 경우(Cut-off)에는 MA 모형을 사용 가능
    - PACF가 특정 차수 이상에서 없어지면 AR 모형을 사용 가능
    - ACF와 PACF 모두 특정 차수 이상에서 없어지는 현상이 나타나지 않는다면 ARMA 모형을 사용
- ARMA 모형인 경우 모수 추정시 AIC/BIC 값을 이용하여 차수 결정 및 모수추정도 동시에 이루어 짐

> **4) ARMA 모형의 모수 추정**
- MM(Method of Modent)/LS(Least Square)/MLE(Maximum Likelihood Estimation) 등의 방법론으로 모수 추정
- ADF(Augmented Dickey Fuller) 검정을 사용하여 해당 수식에 대한 계수 즉 모수 값을 추정
- 부트스트래핑을 사용하여 모수의 표준 오차 추정

> **5) 잔차 진단(모형 진단)**  
: 모형이 추정된 다음 진단(Diagnosis) 과정을 통해 추정이 올바르게 이루어졌는지 검증  
: 기본적으로 잔차(Residual)가 백색 잡음이 되어야 하므로 잔차에 대해 다음을 조사  
- 잔차에 대한 정규성 검정
- 잔차에 대한 ACF 분석 또는 Ljung-Box Q 검정으로 모형 차수 재확인  
: 잔차가 백색잡음이 아니면 새로운 모형으로 위 모든 단계(0\~4단계)를 새로 시작  
: 잔차가 백색잡음이면 일단은 예측력을 확인 -> 예측력이 낮을 시 새로운 모형으로 위 모든 단계(0\~4단계)를 새로 시작  


**3. 분석싸이클 비교:**

| 단계 | 기계학습 | 시계열분석 |
|----------|----------------|--------------------|
| 전처리 | 변수 확장 | 정상성 확보 |
| 시각화 | 모델/변수 선택 | 모델/파라미터 선택 |
| 모델링 | 상동 | 상동 |
| 검증 | 상동 | 상동 |
| 잔차진단 | 상동 | 상동 |

### 분석싸이클 자동화(Non-seasonal)

- **"Hyndman-Khandakar algorithm for automatic ARIMA modelling"**
> 차수가 높지않은 SARIMA 알고리즘을 자동화 한 것으로 [Hyndman-Khandakar 알고리즘(2008)](https://www.jstatsoft.org/article/view/v027i03)을 기반으로 함  
> 정상성변환(Unit Root Calculation), 검증지표 최적화(AIC) 및 MLE 방법을 사용한 모수추정을 모두 포함  

- **자동화 과정:** 일반화 분석싸이클의 2~4단계만 자동화

> **1. KPSS 검정통계량을 사용한 독립변수($Y_t$)의 적분차수/차분차수 결정($0 \leq d \leq 2$)**  

> **2. 차분된 독립변수 $(1 - L)^d Y_t$에 $AIC$를 줄여가며 초기모형 후보들 적합을 통한 Base모형의 차수 $p$와 $q$를 선택**  
> - 만약 $d \leq 1$, 초기모형 후보 5종
    - ARIMA(0,d,0) without constant
    - ARIMA(0,d,0) with constant
    - ARIMA(0,d,1) with constant
    - ARIMA(1,d,0) with constant
    - ARIMA(2,d,2) with constant  
> - 만약 $d = 2$, 초기모형 후보 4종
    - ARIMA(0,d,0) without constant
    - ARIMA(0,d,1) without constant
    - ARIMA(1,d,0) without constant
    - ARIMA(2,d,2) without constant    

> **3. Base모형의 파라미터 튜닝을 통한 Agile모형 선택**  
1) Base모형에서 $p$와 $q$를 $\pm 1$ 변화시키며 AIC들을 추정  
2) $p$와 $q$ 변경 및 상수항(Constant) 반영/미반영하며 AIC들을 추정  
3) 최적의 Agile모형 선택  

> **4. 최종 모형 선택**  
1) 추정된 Agile모형을 Base모형으로 변경  
2) AIC가 더이상 줄어들지 않을 때까지 3번의 과정을 반복하여 Agile모형 재추정  
3) 최종 Agile모형이 최종 선택된 시계열모형  

![TS_Analysis_Cycle](/assets/TS_Analysis_Cycle.png)

~~~
fit = auto_arima(Y_train, stationary=False,
                 trend='c', start_p=0, start_q=0, max_p=5, max_q=5, max_d=3,
                 seasonal=True, start_P=0, start_Q=0, max_P=5, max_Q=5, m=seasonal_diff_order,
                 stepwise=True, trace=True)
~~~

### 실습: 항공사 승객수요 Auro-ARIMA 모델링


```python
# 라이브러리 호출
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
%reload_ext autoreload
%autoreload 2
from module import stationarity_adf_test, stationarity_kpss_test

# 데이터 준비
data = sm.datasets.get_rdataset("AirPassengers")
raw = data.data.copy()

# 데이터 전처리
## 시간 인덱싱
if 'time' in raw.columns:
    raw.index = pd.date_range(start='1/1/1949', periods=len(raw['time']), freq='M')
    del raw['time']

## 정상성 테스트
### 미변환
candidate_none = raw.copy()
display(stationarity_adf_test(candidate_none.values.flatten(), []))
display(stationarity_kpss_test(candidate_none.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_none, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

### 로그 변환
candidate_trend = np.log(raw).copy()
display(stationarity_adf_test(candidate_trend.values.flatten(), []))
display(stationarity_kpss_test(candidate_trend.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_trend, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

trend_diff_order_initial = 0
result = stationarity_adf_test(candidate_trend.values.flatten(), []).T
if result['p-value'].values.flatten() < 0.1:
    trend_diff_order = trend_diff_order_initial
else:
    trend_diff_order = trend_diff_order_initial + 1
print('Trend Difference: ', trend_diff_order)

### 로그+추세차분 변환
candidate_seasonal = candidate_trend.diff(trend_diff_order).dropna().copy()
display(stationarity_adf_test(candidate_seasonal.values.flatten(), []))
display(stationarity_kpss_test(candidate_seasonal.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_seasonal, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()

seasonal_diff_order = sm.tsa.acf(candidate_seasonal)[1:].argmax() + 1
print('Seasonal Difference: ', seasonal_diff_order)

### 로그+추세차분+계절차분 변환
candidate_final = candidate_seasonal.diff(seasonal_diff_order).dropna().copy()
display(stationarity_adf_test(candidate_final.values.flatten(), []))
display(stationarity_kpss_test(candidate_final.values.flatten(), []))
sm.graphics.tsa.plot_acf(candidate_final, lags=100, use_vlines=True)
plt.tight_layout()
plt.show()
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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.82</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.99</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>130.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>996.69</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>1.05</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_97_2](/assets/output_97_2.png)




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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-1.72</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.42</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>130.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-445.40</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>1.05</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_97_5](/assets/output_97_5.png)



    Trend Difference:  1



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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-2.72</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>128.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.48</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-440.36</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>14.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_97_9](/assets/output_97_9.png)



    Seasonal Difference:  12



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
      <th>Stationarity_adf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-4.44</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>118.00</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.49</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>-415.56</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Stationarity_kpss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>0.11</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.10</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>13.00</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




![output_97_13](/assets/output_97_13.png)




```python
## 최종 타겟 선정 및 Train/Test 데이터 분리
candidate = candidate_trend.copy()
split_date = '1958-01-01'
Y_train = candidate[candidate.index < split_date]
Y_test = candidate[candidate.index >= split_date]

## 시각화 및 모수추론(p=1, q=0, d=1, P=1, Q=1, D(m)=12)
plt.figure(figsize=(14,4))
sm.tsa.graphics.plot_acf(Y_train, lags=50, alpha=0.05, use_vlines=True, ax=plt.subplot(121))
sm.tsa.graphics.plot_pacf(Y_train, lags=50, alpha=0.05, use_vlines=True, ax=plt.subplot(122))
plt.show()

# 모델링
## SARIMAX
logarithm, differencing = True, False
# fit_ts_sarimax = sm.tsa.SARIMAX(Y_train, order=(1,trend_diff_order,0), trend='ct').fit()
# fit_ts_sarimax = sm.tsa.SARIMAX(Y_train, order=(1,trend_diff_order,0),
#                                 seasonal_order=(0,0,1,seasonal_diff_order), trend='c').fit()
# fit_ts_sarimax = sm.tsa.SARIMAX(Y_train, order=(1,trend_diff_order,0),
#                                 seasonal_order=(1,0,0,seasonal_diff_order), trend='c').fit()
fit_ts_sarimax = sm.tsa.SARIMAX(Y_train, order=(1,trend_diff_order,0),
                                seasonal_order=(1,0,1,seasonal_diff_order), trend='c').fit()
display(fit_ts_sarimax.summary())
pred_tr_ts_sarimax = fit_ts_sarimax.predict()
pred_te_ts_sarimax = fit_ts_sarimax.get_forecast(len(Y_test)).predicted_mean
pred_te_ts_sarimax_ci = fit_ts_sarimax.get_forecast(len(Y_test)).conf_int()
## 비정상성으로 변환
if logarithm:
    Y_train = np.exp(Y_train).copy()
    Y_test = np.exp(Y_test).copy()
    pred_tr_ts_sarimax = np.exp(pred_tr_ts_sarimax).copy()
    pred_te_ts_sarimax = np.exp(pred_te_ts_sarimax).copy()
    pred_te_ts_sarimax_ci = np.exp(pred_te_ts_sarimax_ci).copy()
if differencing:
    pred_tr_ts_sarimax = np.cumsum(pred_tr_ts_sarimax).copy()

# 검증
%reload_ext autoreload
%autoreload 2
from module import *
Score_ts_sarimax, Resid_tr_ts_sarimax, Resid_te_ts_sarimax = evaluation_trte(Y_train, pred_tr_ts_sarimax,
                                                                             Y_test, pred_te_ts_sarimax, graph_on=True)
display(Score_ts_sarimax)
ax = pd.DataFrame(Y_test).plot(figsize=(12,4))
pd.DataFrame(pred_te_ts_sarimax, index=Y_test.index, columns=['prediction']).plot(kind='line',
                                                                           xlim=(Y_test.index.min(),Y_test.index.max()),
                                                                           linewidth=3, fontsize=20, ax=ax)
ax.fill_between(pd.DataFrame(pred_te_ts_sarimax_ci, index=Y_test.index).index,
                pd.DataFrame(pred_te_ts_sarimax_ci, index=Y_test.index).iloc[:,0],
                pd.DataFrame(pred_te_ts_sarimax_ci, index=Y_test.index).iloc[:,1], color='k', alpha=0.15)
plt.show()

# 잔차진단
error_analysis(Resid_tr_ts_sarimax, ['Error'], Y_train, graph_on=True)
```



![output_98_0](/assets/output_98_0.png)




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>                 <td>value</td>              <th>  No. Observations:  </th>    <td>108</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(1, 1, 0)x(1, 0, [1], 12)</td> <th>  Log Likelihood     </th>  <td>184.253</td>
</tr>
<tr>
  <th>Date:</th>                    <td>Thu, 01 Oct 2020</td>         <th>  AIC                </th> <td>-358.506</td>
</tr>
<tr>
  <th>Time:</th>                        <td>17:36:02</td>             <th>  BIC                </th> <td>-345.142</td>
</tr>
<tr>
  <th>Sample:</th>                     <td>01-31-1949</td>            <th>  HQIC               </th> <td>-353.089</td>
</tr>
<tr>
  <th></th>                           <td>- 12-31-1957</td>           <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>                <td>opg</td>               <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.0001</td> <td>    0.000</td> <td>    0.330</td> <td> 0.741</td> <td>   -0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>   -0.2677</td> <td>    0.080</td> <td>   -3.353</td> <td> 0.001</td> <td>   -0.424</td> <td>   -0.111</td>
</tr>
<tr>
  <th>ar.S.L12</th>  <td>    0.9907</td> <td>    0.008</td> <td>  120.903</td> <td> 0.000</td> <td>    0.975</td> <td>    1.007</td>
</tr>
<tr>
  <th>ma.S.L12</th>  <td>   -0.6113</td> <td>    0.114</td> <td>   -5.341</td> <td> 0.000</td> <td>   -0.836</td> <td>   -0.387</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0014</td> <td>    0.000</td> <td>    7.243</td> <td> 0.000</td> <td>    0.001</td> <td>    0.002</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (Q):</th>          <td>40.18</td> <th>  Jarque-Bera (JB):  </th> <td>0.23</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.46</td>  <th>  Prob(JB):          </th> <td>0.89</td>
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.39</td>  <th>  Skew:              </th> <td>0.09</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.01</td>  <th>  Kurtosis:          </th> <td>3.15</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



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
      <th>MAE</th>
      <th>MSE</th>
      <th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train</th>
      <td>8.27</td>
      <td>197.76</td>
      <td>4.53</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>26.96</td>
      <td>911.04</td>
      <td>6.63</td>
    </tr>
  </tbody>
</table>
</div>




![output_98_3](/assets/output_98_3.png)





![output_98_4](/assets/output_98_4.png)





![output_98_5](/assets/output_98_5.png)



    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.
    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.
    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.
    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.





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
      <th>Stationarity_adf</th>
      <th>Stationarity_kpss</th>
      <th>Normality</th>
      <th>Autocorr(lag1)</th>
      <th>Autocorr(lag5)</th>
      <th>Autocorr(lag10)</th>
      <th>Autocorr(lag50)</th>
      <th>Heteroscedasticity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Test Statistics</th>
      <td>-7.45</td>
      <td>0.33</td>
      <td>0.69</td>
      <td>0.34</td>
      <td>4.17</td>
      <td>12.70</td>
      <td>29.56</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>p-value</th>
      <td>0.00</td>
      <td>0.10</td>
      <td>0.00</td>
      <td>0.56</td>
      <td>0.53</td>
      <td>0.24</td>
      <td>0.99</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Used Lag</th>
      <td>3.00</td>
      <td>13.00</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Used Observations</th>
      <td>104.00</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Critical Value(1%)</th>
      <td>-3.49</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Maximum Information Criteria</th>
      <td>677.41</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Critical Value(10%)</th>
      <td>nan</td>
      <td>0.35</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Alternative</th>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>two-sided</td>
    </tr>
  </tbody>
</table>
</div>





![output_98_8](/assets/output_98_8.png)





![output_98_9](/assets/output_98_9.png)





![output_98_10](/assets/output_98_10.png)





![output_98_11](/assets/output_98_11.png)




```python
# ## 최종 타겟 선정 및 Train/Test 데이터 분리
# candidate = candidate_trend.copy()
# split_date = '1958-01-01'
# Y_train = candidate[candidate.index < split_date]
# Y_test = candidate[candidate.index >= split_date]

# ## 시각화 및 모수추론(p=1, q=0, d=1, P=1, Q=1, D(m)=12)
# plt.figure(figsize=(14,4))
# sm.tsa.graphics.plot_acf(Y_train, lags=50, alpha=0.05, use_vlines=True, ax=plt.subplot(121))
# sm.tsa.graphics.plot_pacf(Y_train, lags=50, alpha=0.05, use_vlines=True, ax=plt.subplot(122))
# plt.show()

# # 모델링
# ## Auto-ARIMA
# logarithm, differencing = True, False
# fit_ts_autoarima = auto_arima(Y_train, stationary=False,
#                               trend='c', start_p=0, start_q=0, max_p=5, max_q=5, max_d=3,
#                               seasonal=True, start_P=0, start_Q=0, max_P=5, max_Q=5, m=seasonal_diff_order,
#                               stepwise=True, trace=True)
# display(fit_ts_autoarima.summary())
# pred_tr_ts_autoarima = fit_ts_autoarima.predict_in_sample()
# pred_te_ts_autoarima = fit_ts_autoarima.predict(len(Y_test), return_conf_int=True)[0]
# pred_te_ts_autoarima_ci = fit_ts_autoarima.predict(len(Y_test), return_conf_int=True)[1]
# ## 비정상성으로 변환
# if logarithm:
#     Y_train = np.exp(Y_train).copy()
#     Y_test = np.exp(Y_test).copy()
#     pred_tr_ts_autoarima = np.exp(pred_tr_ts_autoarima).copy()
#     pred_te_ts_autoarima = np.exp(pred_te_ts_autoarima).copy()
#     pred_te_ts_autoarima_ci = np.exp(pred_te_ts_autoarima_ci).copy()
# if differencing:
#     pred_tr_ts_autoarima = np.cumsum(pred_tr_ts_autoarima).copy()

# # 검증
# %reload_ext autoreload
# %autoreload 2
# from module import *
# Score_ts_autoarima, Resid_tr_ts_autoarima, Resid_te_ts_autoarima = evaluation_trte(Y_train, pred_tr_ts_autoarima,
#                                                                              Y_test, pred_te_ts_autoarima, graph_on=True)
# display(Score_ts_autoarima)
# ax = pd.DataFrame(Y_test).plot(figsize=(12,4))
# pd.DataFrame(pred_te_ts_autoarima, index=Y_test.index, columns=['prediction']).plot(kind='line',
#                                                                            xlim=(Y_test.index.min(),Y_test.index.max()),
#                                                                            linewidth=3, fontsize=20, ax=ax)
# ax.fill_between(pd.DataFrame(pred_te_ts_autoarima_ci, index=Y_test.index).index,
#                 pd.DataFrame(pred_te_ts_autoarima_ci, index=Y_test.index).iloc[:,0],
#                 pd.DataFrame(pred_te_ts_autoarima_ci, index=Y_test.index).iloc[:,1], color='k', alpha=0.15)
# plt.show()

# # 잔차진단
# error_analysis(Resid_tr_ts_autoarima, ['Error'], Y_train, graph_on=True)

```


```python

```
