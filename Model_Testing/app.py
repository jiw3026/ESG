
import os
import re
import math
import requests
import json
import itertools

import numpy as np
import pandas as pd

import onnxruntime
import onnx
import gradio as gr

from huggingface_hub import hf_hub_url, cached_download
from transformers import AutoTokenizer
from transformers import pipeline

try:
    from extractnet import Extractor
    EXTRACTOR_NET = 'extractnet'
except ImportError:
    try:
        from dragnet import extract_content
        EXTRACTOR_NET = 'dragnet'
    except ImportError:
        try:
            import trafilatura
            from trafilatura.settings import use_config
            EXTRACTOR_NET = 'trafilatura'
            trafilatura_config = use_config()
            trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")    #To avoid it runnig signals to avoid clashing with gradio threads
        except ImportError:
            raise ImportError

print('[i] Using',EXTRACTOR_NET)

import spacy

from bertopic import BERTopic

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from unicodedata import normalize



OUT_HEADERS = ['E','S','G']
DF_SP500 = pd.read_csv('SP500_constituents.zip',compression=dict(method='zip'))

MODEL_TRANSFORMER_BASED = "distilbert-base-uncased"
MODEL_ONNX_FNAME = "ESG_classifier_batch.onnx"
MODEL_SENTIMENT_ANALYSIS = "ProsusAI/finbert"
#MODEL3
#BERTOPIC_REPO_ID = "oMateos2020/BERTopic-paraphrase-MiniLM-L3-v2-51topics-guided-model3"
#BERTOPIC_FILENAME = "BERTopic-paraphrase-MiniLM-L3-v2-51topics-guided-model3"
#bertopic_model = BERTopic.load(cached_download(hf_hub_url(BERTOPIC_REPO_ID , BERTOPIC_FILENAME )), embedding_model="paraphrase-MiniLM-L3-v2")

BERTOPIC_REPO_ID = "oMateos2020/BERTopic-distilbert-base-nli-mean-tokens"
BERTOPIC_FILENAME = "BERTopic-distilbert-base-nli-mean-tokens"
bertopic_model = BERTopic.load(cached_download(hf_hub_url(BERTOPIC_REPO_ID , BERTOPIC_FILENAME )))

#SECTOR_LIST = list(DF_SP500.Sector.unique())
SECTOR_LIST = ['Industry',
               'Health',
               'Technology',
               'Communication',
               'Consumer Staples',
               'Consumer Discretionary',
               'Utilities',
               'Financials',
               'Materials',
               'Real Estate',
               'Energy']


def _topic_sanitize_word(text):
    """Función realiza una primera limpieza-normalización del texto a traves de expresiones regex"""
    text = re.sub(r'@[\w_]+|#[\w_]+|https?://[\w_./]+', '', text) # Elimina menciones y URL, esto sería más para Tweets pero por si hay alguna mención o URL al ser criticas web   
    text = re.sub('\S*@\S*\s?', '', text) # Elimina correos electronicos
    text = re.sub(r'\((\d+)\)', '', text) #Elimina numeros entre parentesis
    text = re.sub(r'^\d+', '', text) #Elimina numeros sueltos
    text = re.sub(r'\n', '', text) #Elimina saltos de linea
    text = re.sub('\s+', ' ', text) # Elimina espacios en blanco adicionales
    text = re.sub(r'[“”]', '', text) # Elimina caracter citas 
    text = re.sub(r'[()]', '', text) # Elimina parentesis
    text = re.sub('\.', '', text) # Elimina punto
    text = re.sub('\,', '', text) # Elimina coma
    text = re.sub('’s', '', text) # Elimina posesivos
    #text = re.sub(r'-+', '', text) # Quita guiones para unir palabras compuestas (normalizaría algunos casos, exmujer y ex-mujer, todos a exmujer)
    text = re.sub(r'\.{3}', ' ', text) # Reemplaza puntos suspensivos
    # Esta exp regular se ha incluido "a mano" tras ver que era necesaria para algunos ejemplos
    text = re.sub(r"([\.\?])", r"\1 ", text) # Introduce espacio despues de punto e interrogacion
    # -> NFD (Normalization Form Canonical Decomposition) y eliminar diacríticos
    text = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
                  normalize( "NFD", text), 0, re.I) # Eliminación de diacriticos (acentos y variantes puntuadas de caracteres por su forma simple excepto la 'ñ')
    # -> NFC (Normalization Form Canonical Composition)
    text = normalize( 'NFC', text)

    return text.lower().strip()

def _topic_clean_text(text, lemmatize=True, stem=True):
  words = text.split() 
  non_stopwords = [word for word in words if word not in stopwords.words('english')]
  clean_text = [_topic_sanitize_word(word) for word in non_stopwords] 
  if lemmatize:
    lemmatizer = WordNetLemmatizer()
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text]
  if stem:
    ps =PorterStemmer()
    clean_text = [ps.stem(word) for word in clean_text]

  return ' '.join(clean_text).strip()

SECTOR_TOPICS = []
for sector in SECTOR_LIST:
  topics, _ = bertopic_model.find_topics(_topic_clean_text(sector), top_n=5)
  SECTOR_TOPICS.append(topics)

def _topic2sector(pred_topics):
  out = []
  for pred_topic in pred_topics:
    relevant_sectors = []
    for i in range(len(SECTOR_LIST)):
      if pred_topic in SECTOR_TOPICS[i]:
        relevant_sectors.append(list(DF_SP500.Sector.unique())[i])
    out.append(relevant_sectors)
  return out

def _inference_topic_match(text):
  out, _ = bertopic_model.transform([_topic_clean_text(t) for t in text])
  return out

def get_company_sectors(extracted_names, threshold=0.95):
  '''
  '''
  from thefuzz import process, fuzz
  output = []
  standard_names_tuples = []
  for extracted_name in extracted_names:
    name_match = process.extractOne(extracted_name,
                                            DF_SP500.Name, 
                                            scorer=fuzz.token_set_ratio)
    similarity = name_match[1]/100
    if similarity >= threshold:
      standard_names_tuples.append(name_match[:2])
  
  for extracted_name in extracted_names:
    name_match = process.extractOne(extracted_name,
                                            DF_SP500.Symbol, 
                                            scorer=fuzz.token_set_ratio)
    similarity = name_match[1]/100
    if similarity >= threshold:
      standard_names_tuples.append(name_match[:2]) 

  for std_comp_name, _ in standard_names_tuples:
    sectors = list(DF_SP500[['Name','Sector','Symbol']].where( (DF_SP500.Name == std_comp_name) | (DF_SP500.Symbol == std_comp_name)).dropna().itertuples(index=False, name=None))
    output += sectors
  return output

def filter_spans(spans, keep_longest=True):
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.
    spans (Iterable[Span]): The spans to filter.
    keep_longest (bool): Specify whether to keep longer or shorter spans.
    RETURNS (List[Span]): The filtered spans.
    """
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=keep_longest)
    #print(f'sorted_spans: {sorted_spans}')
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def _inference_ner_spancat(text, limit_outputs=10):
    nlp = spacy.load("en_pipeline")
    out = []
    for doc in nlp.pipe(text):
        spans = doc.spans["sc"]
        #comp_raw_text = dict( sorted( dict(zip([str(x) for x in spans],[float(x)*penalty for x in spans.attrs['scores']])).items(), key=lambda x: x[1], reverse=True) )
        company_list = list(set([str(span).replace('\'s', '').replace('\u2019s','') for span in filter_spans(spans, keep_longest=True)]))[:limit_outputs]
        out.append(get_company_sectors(company_list))
    return out

#def _inference_summary_model_pipeline(text):
#    pipe = pipeline("text2text-generation", model=MODEL_SUMMARY_PEGASUS)
#    return pipe(text,truncation='longest_first')

def _inference_sentiment_model_pipeline(text):
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}#,'return_tensors':'pt'}
    pipe = pipeline("sentiment-analysis", model=MODEL_SENTIMENT_ANALYSIS )
    return pipe(text,**tokenizer_kwargs)

#def _inference_sentiment_model_via_api_query(payload):
#    response = requests.post(API_HF_SENTIMENT_URL , headers={"Authorization": os.environ['hf_api_token']}, json=payload)
#    return response.json()

def _lematise_text(text):
   nlp = spacy.load("en_core_web_sm", disable=['ner'])
   text_out = []
   for doc in nlp.pipe(text): #see https://spacy.io/models#design
       new_text = ""
       for token in doc:
           if (not token.is_punct
               and not token.is_stop
               and not token.like_url
               and not token.is_space
               and not token.like_email
               #and not token.like_num
               and not token.pos_ == "CONJ"):
                    
                new_text = new_text + " " + token.lemma_

       text_out.append( new_text )
   return text_out

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def is_in_archive(url):
    try:
        r = requests.get('http://archive.org/wayback/available?url='+url)
        archive = json.loads(r.text)
    
        if archive['archived_snapshots'] :
            archive['archived_snapshots']['closest']
            return {'archived':archive['archived_snapshots']['closest']['available'], 'url':archive['archived_snapshots']['closest']['url'],'error':0}
        else:
            return {'archived':False, 'url':"", 'error':0}
    except:
        print(f"[E] Quering URL ({url}) from archive.org")
        return {'archived':False, 'url':"", 'error':-1}

#def _inference_ner(text):
#    return labels

def _inference_classifier(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TRANSFORMER_BASED)
    inputs = tokenizer(_lematise_text(text), return_tensors="np", padding="max_length", truncation=True) #this assumes head-only!
    ort_session = onnxruntime.InferenceSession(MODEL_ONNX_FNAME)
    onnx_model = onnx.load(MODEL_ONNX_FNAME)
    onnx.checker.check_model(onnx_model)

    # compute ONNX Runtime output prediction
    ort_outs = ort_session.run(None, input_feed=dict(inputs))

    return sigmoid(ort_outs[0])

def inference(input_batch,isurl,use_archive,filt_companies_topic,limit_companies=10):
    url_list = []    #Only used if isurl
    input_batch_content = []
#    if file_in.name is not "":
#        print("[i] Input is file:",file_in.name)
#        dft = pd.read_csv(
#                file_in.name,
#                compression=dict(method='zip')
#              )
#        assert file_col_name in dft.columns, "Indicated col_name not found in file"
#        input_batch_r = dft[file_col_name].values.tolist()
#    else:
    print("[i] Input is list")
    assert len(input_batch) > 0, "input_batch array is empty"
    input_batch_r = input_batch
 
    print("[i] Input size:",len(input_batch_r))
    
    if isurl:
        print("[i] Data is URL")
        if use_archive:
            print("[i] Use chached URL from archive.org")
        print("[i] Extracting contents using",EXTRACTOR_NET)
        for row_in in input_batch_r:
            if isinstance(row_in , list):
                url = row_in[0]
            else:
                url = row_in
            url_list.append(url)
            if use_archive:
                archive = is_in_archive(url)
                if archive['archived']:
                    url = archive['url']
            #Extract the data from url
            if(EXTRACTOR_NET == 'extractnet'):
              extracted = Extractor().extract(requests.get(url).text)
              input_batch_content.append(extracted['content'])
            elif(EXTRACTOR_NET == 'dragnet'):
              extracted = extract_content(requests.get(url).content)
              input_batch_content.append(extracted)
            elif(EXTRACTOR_NET == 'trafilatura'):
              try:
                  extracted = trafilatura.extract(trafilatura.fetch_url(url), include_comments=False, config=trafilatura_config, include_tables=False)
                  assert len(extracted)>100, "[W] Failed extracting "+url+" retrying with archived version"
              except:
                    archive = is_in_archive(url)
                    if archive['archived']:
                        print("[W] Using archive.org version of",url)
                        url = archive['url']
                        extracted = trafilatura.extract(trafilatura.fetch_url(url), include_comments=False, config=trafilatura_config, include_tables=False)
                    else:
                        print("[E] URL=",url,"not found")
                        extracted = ""
                        url_list.pop() #Remove last from list
                        
              if len(extracted)>100:
                  input_batch_content.append(extracted)
    else:
        print("[i] Data is news contents")
        if isinstance(input_batch_r[0], list):
            print("[i] Data is list of lists format")
            for row_in in input_batch_r:
                input_batch_content.append(row_in[0])
        else:
            print("[i] Data is single list format")
            input_batch_content = input_batch_r
    
    print("[i] Batch size:",len(input_batch_content))
    print("[i] Running ESG classifier inference...")
    prob_outs = _inference_classifier(input_batch_content)
    print("[i] Classifier output shape:",prob_outs.shape)
    print("[i] Running sentiment using",MODEL_SENTIMENT_ANALYSIS ,"inference...")
    sentiment = _inference_sentiment_model_pipeline(input_batch_content )
    print("[i] Running NER using custom spancat inference...")
    ner_labels = _inference_ner_spancat(input_batch_content ,limit_outputs=limit_companies)
    print("[i] Extracting topic using custom BERTopic...")
    topics = _inference_topic_match(input_batch_content)
    news_sectors = _topic2sector(topics)
    
    df = pd.DataFrame(prob_outs,columns =['E','S','G'])
    if isurl:
        df['URL'] = url_list
    else:
        df['content_id'] = range(1, len(input_batch_r)+1)
    df['sent_lbl'] = [d['label'] for d in sentiment ]
    df['sent_score'] = [d['score'] for d in sentiment ]
    df['topic'] = pd.DataFrame(news_sectors).iloc[:, 0]
    #df['sector_pred'] = pd.DataFrame(_topic2sector(topics)).iloc[:, 0] 
    print("[i] Pandas output shape:",df.shape)
    #[[], [('Nvidia', 'Information Technology')], [('Twitter', 'Communication Services'), ('Apple', 'Information Technology')], [], [], [], [], [], []]
    df["company"] = np.nan
    df["sector"] = np.nan
    df["symbol"] = np.nan
    dfo = pd.DataFrame(columns=['E','S','G','URL','sent_lbl','sent_score','topic','company','sector','symbol'])
    for idx in range(len(df.index)):
      if ner_labels[idx]: #not empty
        for ner in ner_labels[idx]:
          if filt_companies_topic:
              if news_sectors[idx]: #not empty
                  if news_sectors[idx][0] not in ner[1]:
                      continue
          dfo = pd.concat( [dfo, df.loc[[idx]].assign(company=ner[0], sector=ner[1], symbol=ner[2])], join='outer', ignore_index=True) #axis=0
    print("[i] Pandas output shape:",dfo.shape)
    return dfo.drop_duplicates()

title = "ESG API Demo"
description = """This is a demonstration of the full ESG pipeline backend where given a list of URL (english, news) the news contents are extracted, using extractnet, and fed to three models:

- A custom scheme for company extraction
- A custom ESG classifier for the ESG labeling of the news
- An off-the-shelf sentiment classification model (ProsusAI/finbert)

API input parameters:
- List: list of text. Either list of Url of the news (english) or list of extracted news contents
- 'Data type': int. 0=list is of extracted news contents, 1=list is of urls.
- `use_archive`: boolean. The model will extract the archived version in archive.org of the url indicated. This is useful with old news and to bypass news behind paywall
- `filter_companies`: boolean. Filter companies by news' topic
- `limit_companies`: integer. Number of found relevant companies to report.

"""
examples = [[ [['https://www.bbc.com/news/uk-62732447'],
            ["https://www.science.org/content/article/suspicions-grow-nanoparticles-pfizer-s-covid-19-vaccine-trigger-rare-allergic-reactions"],
            ["https://www.cnbc.com/2022/09/14/omicron-specific-covid-booster-shot-side-effects-what-to-expect.html"],
            ["https://www.reuters.com/business/healthcare-pharmaceuticals/brazil-approves-pfizer-vaccine-children-young-six-months-2022-09-17/"],
            ["https://www.statnews.com/2022/09/06/pfizer-covid-vaccines-researchers-next-gen-studies/"],
            ["https://www.cms.gov/newsroom/news-alert/updated-covid-19-vaccines-providing-protection-against-omicron-variant-available-no-cost"],
            ["https://www.bbc.com/news/health-62691102"],
            ["https://news.bloomberglaw.com/esg/abbvie-board-faces-new-investor-suit-over-humira-kickback-claims"],
            ["https://esgnews.com/amazon-backed-infinium-to-provide-ultra-low-carbon-electrofuels-for-use-in-trucking-fleet-in-2023/"],
            ["https://esgnews.com/comcast-announces-plan-to-double-energy-efficiency-by-2030-to-power-a-greener-internet/"],
            ["https://esgnews.com/ges-facts-technology-helps-the-city-of-los-angeles-move-closer-to-its-renewable-energy-goals/"],
            ['https://www.bbc.com/news/science-environment-62758811'],
            ['https://www.bbc.com/news/business-62524031'],
            ["https://www.knowesg.com/investors/blackstone-and-sphera-work-together-for-portfolio-decarbonization-program-17022022"],
            ["https://www.esgtoday.com/amazon-partners-with-matt-damons-water-org-to-provide-water-access-to-100-million-people/"],
            ["https://www.esgtoday.com/walmart-allocates-over-1-billion-to-renewable-energy-sustainable-buildings-circular-economy/"],
            ["https://www.esgtoday.com/anglo-american-ties-interest-on-745-million-bond-to-climate-water-job-creation-goals/"],
            ["https://www.esgtoday.com/blackrock-acquires-new-zealand-solar-as-a-service-provider-solarzero/"],
            ["https://www.esgtoday.com/blackrock-strikes-back-against-climate-activism-claims/"],
            ["https://www.esgtoday.com/hm-to-remove-sustainability-labels-from-products-following-investigation-by-regulator/"],
            ["https://www.knowesg.com/sustainable-finance/exxonmobil-fails-the-energy-transition-due-to-failed-governance-structure-04122021"],
            ["https://www.knowesg.com/companies/tesla-is-investigated-by-the-securities-and-exchange-commission-sec-on-solar-07122021"],
            ["https://www.knowesg.com/tech/pcg-and-exxonmobil-will-collaborate-on-plastic-recycling-in-malaysia-20092022"],
            ["https://esgnews.com/nike-launches-community-climate-resilience-program-with-2-million-grant-to-trust-for-public-land/"],
            ["https://esgnews.com/walmart-and-unitedhealth-group-collaborate-to-deliver-access-to-high-quality-affordable-health-care/"],
            ['https://www.bbc.com/news/science-environment-62680423']],'url',False,False,5]]
demo = gr.Interface(fn=inference, 
                    inputs=[gr.Dataframe(label='input batch', col_count=1, datatype='str', type='array', wrap=True),
                            gr.Dropdown(label='data type', choices=['text','url'], type='index', value='url'),
                            gr.Checkbox(label='Parse cached in archive.org'),
                            gr.Checkbox(label='Filter out companies by topic'),
                            gr.Slider(minimum=1, maximum=10, step=1, label='Limit NER output', value=5)],
                    outputs=[gr.Dataframe(label='output raw', col_count=1, type='pandas', wrap=True, header=OUT_HEADERS)],
                             #gr.Label(label='Company'),
                             #gr.Label(label='ESG'),
                             #gr.Label(label='Sentiment'),
                             #gr.Markdown()],
                    title=title,
                    description=description,
                    examples=examples)
demo.launch()
