# Import Libraries
import fitz
from ner import ner_spacy , find_imo,find_swift
import base64
#extract text from pdf
def extract_text(input_file):
    doc = fitz.open(input_file)
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return text

#find entities
def find_ent(input_file):
    text_ents  =[]
    for txt in  extract_text(input_file):
        page_ents = []
        page_ents.extend(ner_spacy(txt)[0]) #ner_spacy(txt)[0]:entities
        page_ents.extend(find_imo(txt))
        page_ents.extend(find_swift(txt))
        text_ents.append(page_ents)
    return text_ents

#highlight entities in pages
def highlight_ent(page , matching_ents):
    for ent in matching_ents:
        matching_val_area = page.search_for(ent.text)
        highlight = page.addHighlightAnnot(matching_val_area)
        info = highlight.info
        info["title"] = ent.label_
        info["content"] = ent.label_
        highlight.set_info(info)
        highlight.update()
    return "highliting done"
#pdf to base64
def pdf_to_base64(pdf):
    with open(pdf, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read())
    return encoded_string
#base64 to pdf
def base64_to_pdf(base64):
    with open('out.pdf', 'wb') as pdf:
        pdf.write(base64.b64decode(base64))
    return ".//out.pdf"
#final function
def output(input_file):
    doc=fitz.open(input_file)
    text_ents = find_ent(input_file)
    i=0
    for page in doc:
        highlight_ent(page,text_ents[i])
        i+=1
    doc.save("output.pdf", garbage=4, deflate=True, clean=True)
    return text_ents,".//ouput.pdf"