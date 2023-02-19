import streamlit as st
import pandas as pd
st.set_page_config(page_title="기업 비교", page_icon="📈")


st.button('SK 케미칼 ESG Score')

st.title('SK 케미칼')

st.title('Our ESG Score')
data=pd.read_excel('kpmg/skchem.xlsx',index_col=0)
st.table(data)

st.title('ESG 평가원 점수')
data=pd.read_excel('kpmg/skchem2.xlsx',index_col=0)
st.table(data)


