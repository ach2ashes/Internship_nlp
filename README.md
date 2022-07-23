
# NER-API
Ce package permet le traitemnt du texte , l'extraction des entités(inclus code swift et code imo), ainsi que le highlighting des ces entités présente dans un fichier pdf




## Installation 
```batch
pip install nerforpdf
```


## Usage/Exemples

```python
import nerforpdf as nfp
nerforpdf.text_preprocessing.text_preprocessing(text,accented=True,stopw=True,punctuation=True,lowercase=True,lemmatize=True,spelling=True,expand_contraction=True,urls=True)
```
cette fonction permet de traiter le text en utilisant les foltres présents comme argument
```python
import nerforpdf as nfp
nerforpdf.text_preprocessing.spacy_preprocessing(text,lowercase=True,stopw=True,punctuation=True,alphabetic=True,lemmatize=True,)
```
Permet de faire du traitement du texte à l'aide de spacy 

```python
import nerforpdf as nfp
nerforpdf.highlight_pdf.output(input_file)

```
cette fonction prend en argument le chemin vers un fichier pdf , extrait les entités(code swift et imo inclus),les highlight , et enregistre le pdf highlighté dans le dossier courant sous le nom "output.pdf"


## API Reference

#### get_entities(text)
Prend un texte(String) et retourne ses entités
#### highlight_pdf(pdf)
Prend le pdf encodé en base64 et retourne le pdf highlighté encodé en base64 ainsi que les entités détectées sous format :    # positions format: a dict containing entities as keys, and the values a list of tuples:(list of Rect positions,page number)
    