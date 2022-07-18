from text_preprocessing import spacy_preprocessing, detect
import spacy

def ner_spacy(text):
    if detect(text) == "en":
        ner  = spacy.load("en_core_web_sm",disable=["tagger","parser"])
    else:
        ner  = spacy.load("fr_core_news_sm",disable=["tagger","parser"])
    labels = ner.get_pipe("ner").labels
    return ner(text).ents,labels




def find_swift(text):
    if detect(text) == "en":
        nlp  = spacy.load("en_core_web_sm",disable=["tagger","parser","ner"])
    else:
        nlp  = spacy.load("fr_core_news_sm",disable=["tagger","parser","ner"])
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler",after="ner")
    else:
        ruler= nlp.get_pipe("entity_ruler")
    text = spacy_preprocessing(text)
    # we use spacy s entity ruler for matching
    patterns =[
        ## uppercases will be lower after preprocessing
         #matches :code(optional) all words wich their lower case form is swift followed  by  space or punct and the code
        {"label":"Code Swift","pattern":[{"LOWER":"code","OP":"?"},{"LOWER":"swift"},{'IS_SPACE':True,'OP':'?'},{"IS_PUNCT":True,"OP":"?"},{"TEXT": {"REGEX": "[A-Za-z0-9]{8,11}"}}]},
        #matches : Swift(e)(with some possible typos) tied to the code
        {"label":"Code Swift","pattern":[{"TEXT": {"REGEX": "(swift)[e]?[A-Za-z0-9]{8,11}"}}]},
        #matches abbreviations:
        {"label":"Code Swift","pattern":[{"TEXT": {"REGEX": "^(sft|st)[e]?[A-Za-z0-9]{8,11}"}}]},
        # matches swift(any special character)code
        {"label":"Code Swift","pattern":[{"TEXT": {"REGEX": "^(swift|sft|st)[e]?[!@#$%^&*()_+\-=\[\]{};':'\\|,.<>\/?][A-Za-z0-9]{8,11}"}}]}
        ]
    ruler.add_patterns(patterns)
    tokens = nlp(text)
    return tokens.ents


def check_imo_funct(imo_code):
    if len(str(imo_code))!=7:
        return False
    else:
        digits = [int(x) for x in str(imo_code)]
        s=0
        for i in range(len(digits)-1):
            s+= digits[i]*(len(digits)-i)
        if s%10==digits[6]:
            return True
        else :
            return False

def find_imo(text):
    if detect(text) == "en":
        nlp = spacy.load("en_core_web_sm",disable=["tagger","parser","ner"])
    else:
        nlp  = spacy.load("fr_core_news_sm",disable=["tagger","parser","ner"])    
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler")
    else:
        ruler= nlp.get_pipe("entity_ruler")
    text = spacy_preprocessing(text)
    # we use spacy s entity ruler for matching
    patterns =[
        #matches : imo 2313... or imo : 2313... ...
        {"label":"Code IMO","pattern":[{"LOWER":"code","OP":"?"},{"LOWER":"imo"},{'IS_SPACE':True,'OP':'?'},{"IS_PUNCT":True,"OP":"?"},{"TEXT": {"REGEX": "[0-9]{7}"}}]},
        #matches imo2343..., also 2313...(alone)
        {"label":"Code IMO","pattern":[{"TEXT": {"REGEX": "^(IMO|imo)?[0-9]{7}"}}]},
         # matches imo(possible typo)(any special character)code
        {"label":"Code IMO","pattern":[{"TEXT": {"REGEX": "^(imo)[a-z]?[!@#$%^&*()_+\-=\[\]{};':'\\|,.<>\/?][0-9]{7}"}}]}
        ]
    ruler.add_patterns(patterns)
    tokens = nlp(text)
    l=[ent[-1] for ent in tokens.ents]
    # we try to extract only the number to check if its really an imo
    possible_imos={}
    for ent,elt in zip(tokens.ents,l) :
        num= str(elt)[-7:]
        possible_imos[ent]=elt
    #check if imo:
    imos = []
    for ent,num in possible_imos.items():
        if check_imo_funct(num):
            imos.append(ent)
    return imos