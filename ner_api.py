from fastapi import FastAPI , HTTPException
from pygments import highlight
from text_preprocessing import text_preprocessing 
from ner import ner_spacy,find_imo,find_swift
from highlight_pdf import output,base64_to_pdf,pdf_to_base64
app = FastAPI()
# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to NER FastAPI"}

@app.get("/entities")
def get_entites(text):
    """
    Text is the text from wich we extract the entities
    """ 
    preprocessed_text = text_preprocessing(text)
    ents = ner_spacy(preprocessed_text)[0]
    ents.extends(find_imo(text))
    ents.extends(find_swift(text))
    return {"text":text,"entities":[(ent.text,ent.label_) for ent in ents]}
@app.get("/highlighted_pdf")
def highlight_pdf(pdf:str):
    pdf =base64_to_pdf(pdf)
    highlighted_pdf,positions = output(pdf)
    highlighted_pdf_base64 = pdf_to_base64(highlighted_pdf)
    # positions format: a dict containing entities as keys, and the values a list of tuples:(list of Rect positions,page number)
    return {"highlighted pdf":highlighted_pdf_base64,"entities":positions}


