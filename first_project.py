import streamlit as st
st.title('C382006 김동욱')
st.write('# 금융계량경제학 프로젝트 구성')

st.write('''
         ## 주제 : :red[SVAR] 이용한 :orange[경제성장]이 각종 지표에 미치는 영향  :sunglasses:
         ''')

# 유튜브 embed URL에 start 파라미터를 붙임 (454초 = 7분 34초)
yt_url = yt_url = 'https://www.youtube.com/embed/u3g1RUW_wL8?start=454&autoplay=1&controls=1'


st.video(yt_url)

with st.container():
    st.markdown('''AI가 요약한 전문(일부수정)       :     지금 이 토론은 사실상 1대 3입니다 새 후보 무조건 성장을 외치고 있습니다.                 하지만 저는 다르게 말하겠습니다저 권영국은 오늘이 자리에서 :grey[**불평등**] 타파를 말하겠습니다 지금 대한민국은 세계 경제 10위 권 입니다 1인당 국민 소득이 35,000달러를 넘어섰습니다 맞습니다. 이 나라의 부는 넘치도록 쌓였습니다 그런데 왜 절반의 국민은 카드값을 걱정하고 청년은 취업 대신 이민을 검색하며 노인은 왜 폐지를 주어야 합니까 돈은 위로 쌓였고 고통은 아래로 홀릅니다 성장은 숫자였을뿐 삶은 나아지지 않았습니다 이대로는 안 됩니다 성장에 가려진 불평등을 직시해야 합니다 해답은 분명합니다 :red[부자 감세가 아니라 부제 증세]여야 합니다 대기업과 고소득자에게 공정한 책임을 묻고 그 재원을 국민에게 되돌려 주어야 합니다 일하는 사람들에게 정당한 대가와 사회 안전망을 제공하겠습니다 
''')

st.info('권영국 후보의 발언에서 몇가지 의문이 생김 ❓❓❓')
st.warning('1. 지금 한국 경제는 충분히 성장한 것인가?')
st.warning('2. 과도한 분배로 인해 경제 성장이 둔화되면 출산율, 부의 편중, 물가 등등 각종 거시 지표에 악영향을 미치지 않을까?')
st.warning('3. 위의 발언에서 언급한 각종 사회문제들은 경기가 둔화되면서 나타나는 문제 아닌가? 하는 의문문')

st.error('결론 : 경제 성장 충격이 다른 각종 변수에 미치는 영향을 분석하고자 한다. ')
st.write(''' #### 변수 설정
        - 기간: (1997.Q1 ~ 2019.Q4) 
            - GDP per capita : 1인당 실질 GDP 
            - 소득 5분위 배율(income_gap) : 소득 5분위별 단위 데이터
            - *출산율*(birth) : 출산율 분기별 데이터
         
        ## 분석과정 : 
            1. EDA : 변수 시각화, 각종 기초 통계량
            2. 데이터 전처리 : 단위근 검정 -> 차분후 분석 진행 
            3. 분석 실행 : VAR 검정 
            4. 분석 해석 
         ''')




st.write('### 데이터 전처리 : 데이터 생성')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests


with st.echo():
    # ✅ GitHub에서 엑셀 파일을 불러오기 위한 **Raw URL** 사용
    url1 = 'https://raw.githubusercontent.com/kimdongug191/nbviewer/main/gg.xlsx'
    
    # ✅ GitHub에서 파일을 가져와서 BytesIO로 변환 후 읽기
    response = requests.get(url1)
    file_bytes = BytesIO(response.content)

    # ✅ pandas를 사용해 엑셀 파일 읽기
    df1 = pd.read_excel(file_bytes, engine='openpyxl')

    # ✅ 데이터 전처리
    df1 = df1.dropna()  # 결측치 제거
    df1 = df1.reset_index(drop=True)  # 인덱스 리셋 (불필요한 index 제거)

    # ✅ Streamlit에서 데이터 출력
    st.write("📊 전처리된 데이터 미리보기")
    st.dataframe(df1.head())

    # 분기별 데이터 생성
    quarters = pd.date_range(start= '1997Q1', end ='2020Q1', freq='QE')
    df2 = pd.DataFrame({
     "Date": quarters,  # 날짜 데이터
     "Quarter": [f"{y}Q{q}" for y, q in zip(quarters.year, quarters.quarter)]})

    df = pd.concat([df1, df2], axis=1, join='inner')    
    df
    df.head() # 사용하면 streamlit 에는 프린트가 안되고 터미널에 출력됨


st.write('### 데이터 전처리 : 시각화 및 단위근 검정')

# 그래프 생성
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].plot(df['Date'], df['GDP per capita'], c='r', lw=0.7)
axes[1].plot(df['Date'], df['income_gap'], c='g', lw=0.7)
axes[2].plot(df['Date'], df['brith'], c='b', lw=0.7)

axes[0].set_title('GDP per capita')
axes[1].set_title('income_gap')
axes[2].set_title('brith')

plt.tight_layout()

# Streamlit에 출력
st.pyplot(fig)

st.write('GDP 데이터는 이미 차분되어 증가율로 변환된 데이터 ')
st.write('income_gap 단위근을 가지는 것을 보인다 ')
st.write('출생률 데이터는 단위근을 가지는 것처럼 보인다.')


with st.echo():
    from statsmodels.tsa.stattools import adfuller


# ADF 검정 함수
    # adfuller(data, lags , type)

    result1 = adfuller(df['GDP per capita'], autolag='AIC' ,regression='ct')
     
    st.write('GDP per capita')
    st.write(f'ADF Statistic: {result1[0]}')
    st.write(f'p-value: {result1[1]}')
    st.write(f'Lags Used: {result1[2]}')
    st.write('income_gap')
    result2 = adfuller(df['income_gap'], autolag='AIC' ,regression='ct')
    st.write(f'ADF Statistic: {result2[0]}')
    st.write(f'p-value: {result2[1]}')
    st.write(f'Lags Used: {result2[2]}')
    
    st.write('brith')
    result3 = adfuller(df['brith'], autolag='AIC' ,regression='ct')
    st.write(f'ADF Statistic: {result3[0]}')
    st.write(f'p-value: {result3[1]}')
    st.write(f'Lags Used: {result3[2]}')
    


st.markdown('단위근 검정 결과 : GDP를 데외한 다른데이터 모두 단위근이 존재 ')
st.markdown('단위근이 존재하는 데이터들을 차분해 분석을 진행 ')

with st.echo() :
    import numpy as np
    df['ln_income'] = np.log(df['income_gap'])
    df['ln_birth'] = np.log(df['brith'])

    df['D_income_gap'] = (df['ln_income'].diff()) * 100
    df['D_birth'] = (df['ln_birth'].diff()) * 100

    df_Test = df.dropna()

    # 그래프 생성
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].plot(df['Date'], df['GDP per capita'], c='r', lw=0.7)
    axes[1].plot(df['Date'], df['D_income_gap'], c='g', lw=0.7)
    axes[2].plot(df['Date'], df['D_birth'], c='b', lw=0.7)

    axes[0].set_title('GDP per capita')
    axes[1].set_title('D_income_gap')
    axes[2].set_title('D_birth')
    plt.tight_layout()

    # Streamlit에 출력
    st.pyplot(fig)

st.markdown('#### 차분데이터를 다시 단위근 검정(이번엔 코드 없이 결과만)')

result1 = adfuller(df_Test['GDP per capita'], autolag='AIC' ,regression='ct')
st.write('##### GDP per capia -> 단위근 없음')
st.write(f'p-value: {result1[1]}')

result2 = adfuller(df_Test['D_income_gap'], autolag='AIC' ,regression='ct')
st.write('##### D_income_gap -> 단위근 없음')
st.write(f'p-value: {result2[1]}')


result3 = adfuller(df_Test['D_birth'], maxlag=3 ,regression='c')
st.write('##### D_birth -> 단위근 없음')
st.write(f'p-value: {result3[1]}')

st.write('### 분석 시작')

with st.echo() :
    df_selected = df[['D_income_gap', 'D_birth', 'GDP per capita']].dropna()

    from statsmodels.tsa.api import VAR

        # VAR 모델 생성 및 적합
    model = VAR(df_selected)
    var_results = model.fit(maxlags=4, ic='aic')

    # 충격 반응 함수(IRF) 생성
    irf = var_results.irf(10)

    # Streamlit 앱 시작
    st.title("충격 반응 함수(IRF) 시각화")

    
    fig = irf.plot(orth=True)  # 'ax=ax' 제거

    st.pyplot(fig)

st.write('Python 처음해보는 프로젝트 및 분석이여서 많은것이 부족함, 하지만 파이썬이 경제학에서 자주사용하는 stata 프로그래밍에 비해 시각화에서 자유로운것이 가장 큰 장점이라고 생각 위 분석은 사실 분석을 한번해본것.. 아주 많은 수정이 필요함 그래도 Python으로 전체 분석을 진행한것에 의미를 두고 코드가 남아있으니 다음번에도 활용가능하고 확장도 가능함')


st.markdown('## 다른 시각화 라이브러리로 시각화 해보기(plotly)')
with st.echo():
    import plotly.express as px
    fig = px.line(df, x='date', y="income_gap", 
              title = 'income_gap', markers =True)
    st.plotly_chart(fig)


    st.markdown(' matplotlib 에서 안되는 :red[facet graph]')
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex",
                 width=800, height=400)

    fig.update_layout(  
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )

    st.plotly_chart(fig)

st.markdown('## 다른 시각화 라이브러리로 시각화 해보기(streamlit 기본)')

with st.echo():
    st.bar_chart(df_selected) # 막대 그래프프

    st.line_chart(df_selected) # 선 그래프

    st.area_chart(df_selected) # 영역 그래프 : 자주 사용할지도?



