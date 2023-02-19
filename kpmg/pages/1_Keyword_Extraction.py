
# Core Pkgs
import streamlit as st 
import os

import pdfminer
from pdfminer.high_level import extract_text
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd
import pandas as pd
import numpy as np
import re, html
from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
from html_table_parser import parser_functions
import itertools
import os

st.set_page_config(page_title="키워드 추출", page_icon="📈")

st.title('주요 키워드 추출') #title
sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
kw_model = KeyBERT(model=sentence_model)

upload_file = st.file_uploader(label='파일을 업로드해주세요')
# defining variables used as input

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

def pdf_to_txt(filename):
    text = extract_text(filename)
    text = text.replace('\n', '')

    return text.split('.')


target=''
if upload_file is not None:
    type=upload_file.type

    if st.button('Run'):
        if 'xml' in type:
            report = st.selectbox('보고싶은 보고서를 클릭해주세요', ['사업개요', '경영의견'])

            com, target = save_chap(upload_file,report)
            target = ' '.join(target)

            st.title(com)
            st.markdown(target)

        if 'pdf' in type:

            target = pdf_to_txt(upload_file)
            st.markdown(target)



    keywords=kw_model.extract_keywords(target, keyphrase_ngram_range=(1, 1), stop_words=None)

    st.metric("Keyword_1", keywords[0][0], keywords[0][1])
    st.metric("Keyword_2", keywords[1][0], keywords[1][1])
    st.metric("Keyword_3",keywords[2][0], keywords[2][1])
    st.metric("Keyword_4", keywords[3][0], keywords[3][1])
    st.metric("Keyword_5", keywords[4][0], keywords[4][1])
