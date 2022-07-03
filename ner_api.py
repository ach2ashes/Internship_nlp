from fastapi import FastAPI , HTTPException
from text_preprocessing import text_preprocessing , ner_spacy

app = FastAPI()
# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to NER FastAPI"}

@app.post("/get_entites")
def get_entites(text):
    """
    Text is the text from wich we extract the entities
    params : a list of boolean to determine the filters we want to activate("accented","stopw","punctuation","lowercase","lemmatize","spelling","expand_contraction","urls")in the order
    """ 
    preprocessed_text = text_preprocessing(text)
    ents = ner_spacy(preprocessed_text)[0]
    return {"text":text,"entities":[(ent.text,ent.label_) for ent in ents]}
