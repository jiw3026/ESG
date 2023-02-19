
# Core Pkgs
import streamlit as st 
import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd


st.set_page_config(page_title="키워드 추출", page_icon="📈")

st.title('Demo: Streamlit model serving and Visualization') #title

# defining variables used as input

data = pd.read_csv('total_report.csv',index_col=0)
data = data.dropna()
data['날짜'] = data['날짜'].astype({'날짜':'int'})
data['사업개요'] = data['사업개요'].astype({'사업개요':'str'})
data['경영의견'] = data['경영의견'].astype({'경영의견':'str'})
company = st.selectbox('회사를 선택해주세요',data.회사명.unique())
temp_data = data[data['회사명']==company]

date = st.selectbox('날짜를 선택해주세요',temp_data.날짜.unique())
temp_data = temp_data[temp_data['날짜']==date]
report = st.selectbox('보고싶은 보고서를 클릭해주세요',['사업개요','경영의견'])
temp_data = temp_data[report].to_string()

clean_text = temp_data.replace('[','')
clean_text = clean_text.replace(']','')
clean_text = clean_text.replace("'","")
clean_text = clean_text.replace(',',' ')

st.markdown(clean_text)


sentence_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
kw_model = KeyBERT(model=sentence_model)

keywords=kw_model.extract_keywords(clean_text, keyphrase_ngram_range=(1, 1), stop_words=None)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Keyword_1", keywords[0][0])
col2.metric("Keyword_2", keywords[1][0])
col3.metric("Keyword_3",keywords[2][0])
col4.metric("Keyword_4", keywords[3][0])
col5.metric("Keyword_5", keywords[4][0])
