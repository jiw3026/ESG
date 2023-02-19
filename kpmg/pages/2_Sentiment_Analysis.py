import pandas as pd
import joblib
import streamlit as st
import sys
from transformers import pipeline
from annotated_text import annotated_text

import pdfminer
from pdfminer.high_level import extract_text
import re, html
from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
from html_table_parser import parser_functions
import itertools
import os

st.set_page_config(page_title="감성분석", page_icon="📈")


# https://huggingface.co/models pre trained models of huggling face ( models)

#https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets ( Streamlit Documentation)
com=''
st.title('ESG 긍/부정 분류') #title
target=''
def pdf_to_txt(filename):
    text = extract_text(filename)
    return text.split('.')
# defining variables used as input
nlp_sa = pipeline('text-classification',model='keonju/deberta_senti')

upload_file = st.file_uploader(label='파일을 업로드해주세요')

def save_chap(data_xml, type='사업개요'):  # 사업의 개요요

    parser_d0350 = SoupStrainer("section-1")
    if type == '사업개요':
        section2_pattern = re.compile(
            r"<SECTION-1((?!<SECTION-1)[\S\s\n])*?(D-0-2-0-0)[\S\s\n]*?</SECTION-1>")  # chap4 -> 0400, #cha3 0300
    else:
        section2_pattern = re.compile(r"<SECTION-1((?!<SECTION-1)[\S\s\n])*?(D-0-4-0-0)[\S\s\n]*?</SECTION-1>")
    find_company = re.compile(r'<COMPANY-NAME AREGCIK="[0-9]+">(.+)</COMPANY-NAME>')

    dsd_xml = data_xml.read()
    try:
        dsd_xml = dsd_xml.decode('utf-8')
        dsd_xml = dsd_xml.replace('&cr;', '&#13;')
        dsd_xml = re.sub('(\n|\r)?</*SPAN.*?>(\n|\r)?', '', dsd_xml)
        dsd_xml = html.unescape(dsd_xml)
        com = find_company.search(dsd_xml).group(0)
        com = re.sub('<COMPANY-NAME AREGCIK="[0-9]+">', '', com)
        com = re.sub('</COMPANY-NAME>', '', com)
        section2_section = section2_pattern.search(dsd_xml)
        section2_section = section2_section.group()
    except:
        dsd_xml = dsd_xml.decode('cp949')
        dsd_xml = dsd_xml.replace('&cr;', '&#13;')
        dsd_xml = re.sub('(\n|\r)?</*SPAN.*?>(\n|\r)?', '', dsd_xml)
        dsd_xml = html.unescape(dsd_xml)
        com = find_company.search(dsd_xml).group(0)
        com = re.sub('<COMPANY-NAME AREGCIK="[0-9]+">', '', com)
        com = re.sub('</COMPANY-NAME>', '', com)
        section2_section = section2_pattern.search(dsd_xml)
        section2_section = section2_section.group()



    if section2_section != None:
        remark_page = BS(section2_section, 'lxml', parse_only=parser_d0350).find("section-1")
        remark_page.find().text
        chap6 = [list(text.stripped_strings) for text in remark_page.find_all(recursive=False)]
        lis = list(itertools.chain.from_iterable(chap6))
    else:
        lis = None

    return com,lis

if upload_file is not None:
    type=upload_file.type

    target2=''
    if 'xml' in type:
        report = st.selectbox('보고싶은 보고서를 클릭해주세요', ['사업개요', '경영의견'])
        if st.button('Run'):
            com, target = save_chap(upload_file,report)

        st.title(com)
        target2 = []
        for i in target:
            c=i.split('.')
            for z in c:
                target2.append(z)

    if 'pdf' in type:
        if st.button('Run'):

            target = pdf_to_txt(upload_file)
            target2 = target

            target2 = list(filter(None, target2))
sentence_result=nlp_sa(target2)
for i in range(len(target2)):
    if sentence_result[i]['label'] == '중립':
        result_text = (target2[i],'중립')
    elif sentence_result[i]['label'] == '긍정':
        result_text = (target2[i],'긍정')
    else:
        result_text = (target2[i],'부정')
    st.write(annotated_text(result_text))