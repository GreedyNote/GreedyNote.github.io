---
title:  "[데이터 수집] 구글 트렌드 api"
excerpt: "구글 트렌드 api 사용법에 대해 알아보자"
toc: true
toc_sticky: true
header:
  teaser: /assets/images/python_logo.jpg

categories:
  - data acquisition
tags:
  - Google trend
---

## 사용법

#### 1. Connect to Google

~~~
from pytrends.request import TrendReq

pytrends = TrendReq()
~~~

#### 2. Build Payload
**Parameter**
- `kw_list`: 검색할 키워드 (필수)
- `timeframe`: 추출할 기간
- `geo`: 로컬 지역

~~~python
kw_list = ["Blockchain"]
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='KR')
~~~

#### 3. Use API methods
- `Interest Over Time`: 시간에 따른 Trend 변화
- `Interest by Region`: 지역별 검색 트렌드 비교
- `suggestions`: 추천 검색어

## 시간에 따른 변화


```python
keyword = ['gme', 'amc']
period = 'today 12-m'

# Google Trend 접속
from pytrends.request import TrendReq
import os
trend_obj = TrendReq()

# Build Payload
trend_obj.build_payload(kw_list=keyword, timeframe=period)

# 시간에 따른 변화
trend_df = trend_obj.interest_over_time()
print(trend_df.head())

# 그래프 출력
plt.style.use('ggplot')
plt.figure(figsize=(14,5))
trend_df['gme'].plot()
trend_df['amc'].plot()
plt.title('Google Trends over time', size=15)
plt.legend()

# 그래프 파일 저장
cwd = os.getcwd()
output_filepath = os.path.join(cwd, "output", "google_trend_{}.png".format(keyword))
plt.savefig(output_filepath, dpi=300)
```

                gme  amc  isPartial
    date                           
    2020-06-07    0    4      False
    2020-06-14    0    5      False
    2020-06-21    0    4      False
    2020-06-28    0    4      False
    2020-07-05    0    4      False




![output_7_1](/assets/output_7_1.png)



## 지역별 트렌드 비교


```python
keyword = "gme"
period = 'today 12-m'

# Google Trend 접속
trend_obj = TrendReq()

# build_payload
trend_obj.build_payload(kw_list=[keyword], timeframe=period)

# 지역별 검색 트렌드 비교
trend_df = trend_obj.interest_by_region()
trend_df.sort_values(by=[keyword], ascending=False, inplace=True)

# 그래프 출력
plt.style.use('ggplot')
plt.figure(figsize=(14,10))
trend_df.iloc[:20][keyword].plot(kind='bar')
plt.title('Google Trends by Region', size=15)
plt.legend()

# 그래프 저장
cwd = os.getcwd()
output_filepath = os.path.join(cwd, "output", "google_trend_{}.png".format(keyword))
plt.savefig(output_filepath, dpi=300)
```


​    
![output_9_0](/assets/output_9_0.png)
​    


## 로컬(한국) 데이터


```python
keyword1 = '삼성'
keyword2 = '애플'
local_area = "KR"
period = 'today 5-y'

# Google Trend 접속
from pytrends.request import TrendReq
trend_obj = TrendReq()

# build payload
trend_obj.build_payload(kw_list=[keyword1, keyword2], timeframe=period, geo=local_area)

# 시간별 트렌드 검색
trend_df = trend_obj.interest_over_time()

# 그래프 출력
plt.style.use('ggplot')
plt.figure(figsize=(14,5))
trend_df[keyword1].plot()
trend_df[keyword2].plot()
plt.title(f'Google Trends {keyword1} vs {keyword2}', size=15)
plt.legend()

# 그래프 저장
cwd = os.getcwd()
output_filepath = os.path.join(cwd, f'Google Trends {keyword1} vs {keyword2}.png')
plt.savefig(output_filepath, dpi=300)
```


​    
![output_11_0](/assets/output_11_0.png)
​    


## 추천 검색어


```python
keyword = 'mugen'

# Google Trend 접속
from pytrends.request import TrendReq
trend_obj = TrendReq()

# 추천 검색어 조회
suggested_keywords = trend_obj.suggestions(keyword)

# 출력
pprint(suggested_keywords)
```

    [{'mid': '/g/11j21njpbx',
      'title': 'Demon Slayer: Kimetsu no Yaiba the Movie: Mugen Train',
      'type': '2020 film'},
     {'mid': '/m/05l9t3', 'title': 'Mugen', 'type': 'Game engine'},
     {'mid': '/m/06lzjz', 'title': 'Mugen Motorsports', 'type': 'Company'},
     {'mid': '/m/0233bn', 'title': 'Infernal Affairs', 'type': '2002 film'},
     {'mid': '/g/11h7hvfmrn',
      'title': 'Scythe SCMG-5100 Mugen 5 Rev.b Cpu Cooler',
      'type': 'Topic'}]



```python

```
