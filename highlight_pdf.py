# Import Libraries
import re
import fitz
from text_preprocessing import ner_spacy

list_of_colors = ["red","blue","green","yellow","cyan","brown","pink","magenta","orange"]
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
def highlight_ent(page , matching_ents,color_map):
    for ent in matching_ents:
        matching_val_area = page.search_for(ent.text)
        highlight = page.addHighlightAnnot(matching_val_area)
        highlight.set_colors(colors= fitz.utils.getColor(color_map[ent.label_]))
        info = highlight.info
        info["title"] = ent.label_
        info["content"] = ent.label_
        highlight.set_info(info)
        highlight.update()
    return "highliting done"
#final function
def output(input_file):
    doc=fitz.open(input_file)
    text = extract_text(input_file)
    text_ents = find_ent(input_file)
    labels = ner_spacy(text[0])[1]
    color_map = dict(zip(labels, list_of_colors[:len(labels)]))
    i=0
    for page in doc:
        highlight_ent(page,text_ents[i],color_map)
        i+=1
    doc.save("output.pdf", garbage=4, deflate=True, clean=True)
    return "updated pdf"