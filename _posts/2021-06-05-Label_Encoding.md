---
title:  "[데이터 전처리] 데이터 인코딩"
excerpt: "범주형 데이터를 숫자형으로 인코딩해보자"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/python_logo.jpg

categories:
  - data preprocessing
tags:
  - data encoding
---
# 데이터 인코딩
머신러닝 알고리즘은 문자열 데이터 속성을 입력받지 않으며 모든 데이터는 숫자형으로 표현되어야 합니다.  
문자형, 카테고리형 속성은 모두 숫자값으로 변환/인코딩 되어야 합니다.

범주형 인코딩을 하기위해서 어떻게 하면될까요? 대표적으로 두가지 방법이 있습니다.
- **Label Encoding**
- **One-Hot Encoding**  

## 라벨인코딩 (Label Encoding)
라벨 인코딩이란 알파벳 오더순으로 숫자를 할당해주는 것을 말합니다.

![5](/assets/5.png)

하지만 라벨인코딩은 적용에 난점이 있습니다. 위의 전자제품 데이터는 순서나 랭크가 없습니다.
그러나 라벨인코딩을 수행하면 결국엔 알파벳 순으로 랭크가 되는 것이고, 그로 인해서 랭크된 숫자정보가 모델에 잘못 반영될 수가 있습니다. 그래서 나온것이 또다른 방법인 One-Hot-Encoding 입니다.

### LabelEncoder
이제부터 실제로 라벨 인코딩을 해보겠습니다. sklearn의 LabelEncoder 클래스를 사용합니다.
- `classes_` : 인코딩된 라벨들을 리스트형태로 출력합니다
- `inverse_transform` : 본래 값을 반환합니다


```python
from sklearn.preprocessing import LabelEncoder
items = ['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값: ', labels)
```

    인코딩 변환값:  [0 1 4 5 3 3 2 2]



```python
print('인코딩 클래스: ', encoder.classes_)
```

    인코딩 클래스:  ['TV' '냉장고' '믹서' '선풍기' '전자렌지' '컴퓨터']



```python
print('디코딩 원본 값: ', encoder.inverse_transform([0,1,4,5,3,3,2,2]))
```

    디코딩 원본 값:  ['TV' '냉장고' '전자렌지' '컴퓨터' '선풍기' '선풍기' '믹서' '믹서']


## 원 핫 인코딩 (One Hot Encoding)
피처값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 컬럼에만 1을 표시하고 나머지 컬럼에는 0을 표시하는 방식입니다.

![6](/assets/6.png)

### get_dummies()
이제부터 실제로 원핫 인코딩을 해보겠습니다. 판다스의 get_dummies()함수를 이용합니다.


```python
df = pd.DataFrame({'item': ['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
pd.get_dummies(df)
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
      <th>item_TV</th>
      <th>item_냉장고</th>
      <th>item_믹서</th>
      <th>item_선풍기</th>
      <th>item_전자렌지</th>
      <th>item_컴퓨터</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
