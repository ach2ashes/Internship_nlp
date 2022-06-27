#import libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
import string
import contractions
from spellchecker import SpellChecker
import re
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
spell = SpellChecker()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def text_preprocessing(text,lang):
    if lang.lower()=='english':
        stopword =stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
    else :#if lang="french"
        stopword = stopwords.words('french')
        lemmatizer = FrenchLefffLemmatizer()
    #lowercase the text 
    lower_text = text.lower()
    #remove urls
    lower_text=remove_urls(lower_text)
    #tokenize the text 
    tokens = WhitespaceTokenizer().tokenize(lower_text)
    #expand contractions
    tokens_expanded = [contractions.fix(token) for token in tokens]
    #remove punctuation
    no_punct = [token for token in tokens_expanded if token not in string.punctuation]
    #remove stopwords
    no_stopwords = [token for token in no_punct if token not in stopword]
    #spell check:
    correct_words = [spell.correction(token) for token in no_stopwords]
    #lemmatization : 
    lemmatized = [lemmatizer.lemmatize(token) for token in correct_words]
    return ' '.join(word for word in lemmatized)

    #Some tests:
en_test = ' 13,000 people receive #wildfires evacuation orders in California'
fr_test = 'plusieurs phrases pour un nettoyage'
print(text_preprocessing(en_test,'english'))
print(text_preprocessing(fr_test,'french'))