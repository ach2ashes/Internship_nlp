#import libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer , TreebankWordTokenizer,WordPunctTokenizer
from nltk.corpus import stopwords
import string
import contractions
from spellchecker import SpellChecker
import re
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import unicodedata
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

def text_preprocessing(text,lang,accented=True,tokenizer=WhitespaceTokenizer(),acccented=True,stopw=True,punctuation=True,lowercase=True,lemmatize=True,spelling=True,expand_contraction=True,urls=True):
    if lang.lower()=='english':
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
    tokens = tokenizer.tokenize(text)
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
    return ' '.join(word for word in tokens)

    #Some tests:
file = 'C:/Users/PC2/Downloads/test-ex.txt'
f=open(file,'r')
data = f.read()
print(text_preprocessing(data,lang='french'))
#Now we ll do preprocessing using mainly spacy
import spacy
#load only french and english models tokenizers
nlp_en = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
nlp_fr = spacy.load("fr_core_news_sm", disable=['parser', 'tagger', 'ner'])

def spacy_preprocessing(text,lang,lowercase=True,stopw=True,punctuation=True,alphabetic=True,lemmatize=True,):
    if lang =="en":
        nlp = nlp_en
    else :
        nlp = nlp_fr
    if lowercase:
        text = text.lower()
    #tokenize with spacy's default tokenizer
    tokens = nlp(text)
    if stopw :
        tokens = [token for token in tokens if not token.is_stop]
    if lemmatize :
        tokens = [token.lemma_.strip() for token in tokens]
    if punctuation :
        tokens = [re.sub('<[^>]*>', '', token) for token in tokens]
    if alphabetic:
        tokens = [re.sub('[\W]+','',token.lower()) for token in tokens]
    return ' '.join(word for word in tokens)

file = 'C:/Users/PC2/Downloads/test-ex.txt'
f=open(file,'r')
data = f.read()
print(spacy_preprocessing(data,lang='french'))