import re
import nltk
nltk.download('stopwords')
import spacy
import unicodedata
from unidecode import unidecode
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer



tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    # Put your code
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    return text


def stem_text(text):
    # Put your code
    porter = PorterStemmer()
    tokens = tokenizer.tokenize(text)
    #convert text in a list of words
    tokens = [token.strip() for token in tokens]
    #new list of stemmed list of words 
    words = []
    for w in tokens:
        w2 = porter.stem(w)
        words.append(w2)
    #get the new stemmed text
    text = " ".join(words)

    # porter = PorterStemmer()
    # return " ".join([porter.stem(word) for word in text.split()])

    return text


def lemmatize_text(text):
    # Put your code
    sentence = nlp(text)
    return " ".join([token.lemma_ for token in sentence])
    

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Put your code
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),flags=re.IGNORECASE|re.DOTALL)
    for key, values in CONTRACTION_MAP.items():
        text = re.sub(key, values, text, flags = re.IGNORECASE|re.DOTALL)
    return text


def remove_accented_chars(text):
    # nfkd= unicodedata.normalize('NFKD', text)
    # ascii = nfkd.encode('ASCII', 'ignore')
    # text = ascii.decode('UTF-8')
    #return text
    return unidecode(text)


def remove_special_chars(text, remove_digits=False):
    # Put your code
    # for w in tokens:
    #     if remove_digits == True:
    #         w2=re.sub('[^A-Za-z0-9.]+', ' ',w)

    # return text

    if remove_digits:
        return "".join([char for char in text if (char.isalpha() or char == ' ')])
    else:
        return "".join([char for char in text if (char.isalnum() or char == ' ')])


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    # Put your code
    clean_words = []
    for word in text.split():
        if word.lower() not in stopwords:
            clean_words.append(word.lower())
    text = " ".join(clean_words)
    return text
    
    #return ' '.join(word.lower() for word in text.split() if word.lower() not in stopwords)
    #return text


def remove_extra_new_lines(text):
    # Put your code
    return str.join(' ', text.splitlines())
   

def remove_extra_whitespace(text):
    # Put your code
    return " ".join(text.split())
    

def normalize_corpus(
    corpus,  #:list[str]
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
