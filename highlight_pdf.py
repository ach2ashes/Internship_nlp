# Import Libraries
import fitz
from text_preprocessing import ner_spacy
from spacy import displacy
from pathlib import Path

#extract text from pdf
def extract_text(input_file):
    doc = fitz.open(input_file)
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return text

#find entities
def find_ent(input_file):
    text_ents  = [ner_spacy(txt)[0] for txt in  extract_text(input_file)] #ner_spacy(txt)[0]:entities
    return text_ents

#highlight entities in pages
def highlight_ent(text):
    html = displacy.render(text, style="ent",jupyter=False,page = True)
    return html
#final function
def output(input_file):
    doc=fitz.open(input_file)
    text = extract_text(input_file)
    text_ents = find_ent(input_file)
    svg=highlight_ent(text_ents)
    output_path = Path("output.html")
    output_path.open("w", encoding="utf-8").write(svg)
    return "created html"

input_file="C:\\Users\\PC2\\Downloads\\yanni.pdf"
output(input_file)   
#convert to pdf 
import pdfkit # to run this we need to install wkhtmltopdf and set up environnement variables
pdfkit.from_file('output.html', 'out.pdf')