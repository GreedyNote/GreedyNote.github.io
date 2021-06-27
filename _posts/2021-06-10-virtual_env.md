---
title:  "파이썬 가상환경"
toc: true
toc_sticky: true

categories:
  - basic syntax
---
## 가상환경을 사용하는 이유
파이썬을 사용하다보면 pip로 패키지를 설치하게 되는데 이 패키지들은 **파이썬 설치 디렉터리**의 `Lib/site-package` 안에 저장됩니다. 그래서 pip로 설치한 패키지는 모든 파이썬 스크립트에서 사용할 수 있게 됩니다. 이런 방식이 큰 문제가 없지만 프로젝트를 여러 개 개발할 때는 패키지의 버전 문제가 발생합니다.  
예를들어 프로젝트 A에서는 버전 1.5를 사용해야하고 프로젝트 B에서는 버전 2.0을 사용해야 하는 경우가 생깁니다. 이 패키지의 버전이 호환되지 않는다면 개발하기가 상당히 불편해집니다.

![가상환경1](/assets/가상환경1.png)

## 가상환경 (virtual environment)
이런 문제를 해결하기 위해 파이썬에서는 가상 환경을 제공하는데, 가상 환경은 독립된 공간을 만들어주는 공간입니다. 가상 환경에서 pip로 패키지를 설치하면 **가상 환경 디렉터리**의 `Lib/site-packages` 안에 패키지를 저장해줍니다. 즉, 프로젝트 A와 B 각각 가상 환경을 만들어서 서로 다른 버전의 패키지를 설치할 수 있습니다.

![가상환경2](/assets/가상환경2.png)

## Windows에서 가상 환경 만들기
윈도우에서 가상 환경을 만드는 방법을 알아보겠습니다. 가상 환경은 venv 모듈에 가상 환경 이름을 지정해서 만듭니다.
```
python -m venv 가상환경이름
```  
<br>

예시로 `C:\project` 폴더 아래에 가상 환경을 만들겠습니다.  
1. 다음과 같이 명령 프롬프트에서 example 가상환경을 만들고
2. example 폴더 안으로 이동합니다.
3. Scripts 폴더 안의 activate.bat 파일을 실행하면 가상 환경이 활성화 됩니다.
```
C:\project>python -m venv .env
C:\project\.env\Scripts\activate.bat
```
가상환경에서 패키지를 설치할때는
```
(.env) C:\project>pip install [패키지]
```
### cd 명령어 (명령 프롬프트)
cd는 디렉토리 변경(change directory)의 줄임말입니다. 디렉토리를 변경할 때 기초가 되는 명령어입니다.
1. 원하는 상위 폴더로 이동 : `cd [폴더명]`
2. 폴더 한단계 아래로 이동 : `cd..`
3. 루트 폴더로 이동 : `cd\`

## requirements.txt 작성
가상환경 설정이 끝났으면 이 가상환경에서 내가 설치한 패키지들을 작성해놓을 파일을 만들어주자
```
pip freeze > requirements.txt
```
내가 담당한 부분을 개발 완료 후, git에 개발한 소스코드를 push하고 나면 통할할 때 해당 소스코드를 pull한 후, 상대방이 명령어를 치게된다면 어려움없이 패키지들을 설치할 수 있다
```
pip install -r requirements.txt
```

## 아나콘다 가상 환경
1. 가상 환경 만들기
```
conda create -n venv_trebo python=3.8
```
<br>

2. 가상 환경 활성화 하기
```
activate venv_trebo
```
<br>

3. 라이브러리 다운받기 (여기서부턴 가상환경 활성화 해야함)
```
pip install [라이브러리명]
```
<br>

4. 라이브러리 버전들을 확인하기
```
conda list
```
<br>
5. conda list 저장하기 (export)
```
conda list --export > packagelist.txt
```

<br>
6. conda list 불러오기 (import)
```
conda install --file packagelist.txt
```
