# Text Preprocessing with NLTK
#import libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer , TreebankWordTokenizer,WordPunctTokenizer,word_tokenize
from nltk.corpus import stopwords
import string
import contractions
from spellchecker import SpellChecker
import re
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import unicodedata
from langdetect import detect
from itertools import chain , combinations
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


def text_preprocessing(text,accented=True,stopw=True,punctuation=True,lowercase=True,lemmatize=True,spelling=True,expand_contraction=True,urls=True):
    if detect(text)=='en':
        stopword =stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        spell = SpellChecker()
    else :#if lang="french"
        stopword = stopwords.words('french')
        lemmatizer = FrenchLefffLemmatizer()
        spell = SpellChecker(language="fr")
    if lowercase:
        #lowercase the text 
        text = text.lower()
    if urls:
        #remove urls
        text=remove_urls(text)
    #tokenize the text 
    tokens =WhitespaceTokenizer().tokenize(text)
    if expand_contraction:
        #expand contractions
        tokens = [contractions.fix(token) for token in tokens]
    if punctuation:
        #remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
    if stopw:
        #remove stopwords
        tokens = [token for token in tokens if token not in stopword]
    if accented:
        tokens = [remove_accented_chars(token) for token in tokens]
    if spelling:
        #spell check:
        tokens = [spell.correction(token) for token in tokens]
    if lemmatize:
        #lemmatization : 
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

    #Some tests:
file = 'C:/Users/PC2/Downloads/test-ex.txt'
f=open(file,'r')
data = f.read()
#print(text_preprocessing(data))


# Text Preprocessing with spaCy

#Now we ll do preprocessing using mainly spacy
import spacy
#load only french and english models tokenizers
nlp_en = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
nlp_fr = spacy.load("fr_core_news_sm", disable=['parser', 'tagger', 'ner'])

def spacy_preprocessing(text,lowercase=True,stopw=True,punctuation=True,alphabetic=True,lemmatize=True,):
    if detect(text)=="en":
        nlp = nlp_en
    else :
        nlp = nlp_fr
    if lowercase:
        text = text.lower()
    remove_accented_chars(text)
    #tokenize with spacy's default tokenizer
    tokens = nlp(text)
    if stopw :
        tokens = [token for token in tokens if not token.is_stop]
    if lemmatize :
        tokens = [token.lemma_.strip() for token in tokens]
    if punctuation :
        tokens = [re.sub('<[^>]*>', '', token) for token in tokens]
    if alphabetic:
        tokens = [re.sub('[\W]+','',token) for token in tokens]
    return ' '.join(word for word in tokens)

file = 'C:/Users/PC2/Downloads/test-ex.txt'
f=open(file,'r')
data = f.read()
#print(spacy_preprocessing(data,lowercase=False))

#Definig NER for nltk and spacy


import spacy


def ner_nltk(text):
    assert detect(text) =='en' 'text should be english to be parsed with nltk'
    tokens = word_tokenize(text)
    tagged_tokens  = nltk.pos_tag(tokens)
    chunked = nltk.ne_chunk(tagged_tokens)
    for chunk in chunked : 
        if hasattr(chunk,"label") and chunk.label == "NE":
            print(chunk)


def ner_spacy(text):
    if detect(text) == "en":
        ner  = spacy.load("en_core_web_sm",disable=["tagger","parser"])
    else:
        ner  = spacy.load("fr_core_news_sm",disable=["tagger","parser"])
    labels = ner.get_pipe("ner").labels
    return ner(text).ents,labels



#Comparing results before\after preprocessing

#Testing preprocessing filters
from itertools import chain , combinations
def powerset(iterable): # return all possible subsets of iterable
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

list_of_params=["accented","stopw","punctuation","lowercase","lemmatize","spelling","expand_contraction","urls"] 
dict_of_params=dict.fromkeys(list_of_params,[True])

def filter_combinations(params):#return all the possible filter combinations, param; list of parametrs with a default value
    list_of_combs = []
    combs=powerset([i for i in range(len(params)) ])
    for comb in combs:
        initial_conf = [True for i in range(len(params))]
        if len(comb)==0 :
            list_of_combs.append(initial_conf)
        else:
            for i in comb:
                initial_conf[i]=False
            list_of_combs.append(initial_conf)
    return list_of_combs

import pandas as pd 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
columns = []
columns.extend(list_of_params)
ff=filter_combinations(list_of_params)#all possible filter combinations
def results(text,preprocessing_function):
    table=[]
    labels = ner_spacy(text)[1]
    columns.extend(labels)
    for comb in ff:#comb is a combinations of active filters 
        preprocessed_text=preprocessing_function(text,*comb)
        entities = ner_spacy(preprocessed_text)[0]
        dictionn = dict.fromkeys(labels,0)
        for ent in entities:
            for key in dictionn.keys():
                if ent.label_==key:
                    dictionn[key]+=1
        comb.extend(dictionn.values())
        table.append(comb)       
    return pd.DataFrame(table,columns=columns)
#french NER in spaCy has only  4 labels





