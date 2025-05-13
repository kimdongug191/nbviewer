import streamlit as st
st.title('첫번째 웹 어플 만들기 😄')

st.write('# H1. Markdown으로 대화하기')
st.write('### H3. Markdown으로 대화하기')
st.write('') # 빈줄 추가

st.header('헤더 : st.header()')
st.subheader('서비헤터 : st.subheader()')
st.text('본문 텍스트 : st.text()')

st.markdown(
    '''
    #마크다운 헤더1
    - 마크다운 목록1. **굵게** 표시
    - 마크다운 목록2. *기울임* 표시
        - 마크다운 목록2-1
        - 마크다운 목록2-2

    ## 마크다운 헤더2
    - [네이버](https://naver.com)
    - [구글](https://google.com)

    ### 마크다운 헤더3
    일반 텍스트 ''')

st.divider() # 구분선선
