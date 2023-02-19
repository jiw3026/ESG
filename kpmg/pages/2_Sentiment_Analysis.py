import pandas as pd
import joblib
import streamlit as st
import sys
from transformers import pipeline
from annotated_text import annotated_text


st.set_page_config(page_title="감성분석", page_icon="📈")


# https://huggingface.co/models pre trained models of huggling face ( models)

#https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets ( Streamlit Documentation)

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

context = st.text_area('선택한 보고서의 전체 내용',clean_text)
text_list = context.split('.')
n_sentence=st.slider(label='몇번째 문장을 선택하시겠습니까?', min_value=0, max_value=len(text_list), step=1, label_visibility="visible")

st.header('please provide necessary inputs for sentiment- analysis')
user_text_input_sentiment = st.text_input('Please type context', text_list[n_sentence])


nlp_sa = pipeline('text-classification',model='keonju/deberta_senti')
sentence_result=nlp_sa(user_text_input_sentiment)
if sentence_result[0]['label'] =='긍정':
    st.write('해당 문장은 긍정입니다.')
if sentence_result[0]['label'] == '중립':
    st.write('해당 문장은 중립입니다.')
result = nlp_sa(text_list)

result_list= []
for i in range(len(text_list)):
    if result[i]['label'] == '2':
        result_text = (text_list[i],'긍정')
    elif result[i]['label'] == '부정':
        result_text = (text_list[i],'부정')
    else:
        result_text = (text_list[i],'중립')
    st.write(annotated_text(result_text))