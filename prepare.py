
import pandas as pd

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def basic_clean(string):
    string = string.lower()
    string = (unicodedata.normalize('NFKD', string)
                         .encode('ascii', 'ignore')
                         .decode('utf-8', 'ignore')
             )
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def clean_html(string):
    string = re.sub(r'<[^>]*>', '', string)
    string = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", '', string)
    string = re.sub(r'\n', '', string)
    string = re.sub(r'\s\s', '', string)
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    return ' '.join(stems)

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    return ' '.join(lemmas)

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    
    for word in extra_words:
        stopword_list.append(word)
    
    for word in exclude_words:
        stopword_list.remove(word)
        
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    return ' '.join(filtered_words)

def prepare_readme_data(df, column):

    df = df.dropna()
    df = df[df['language'].map(df['language'].value_counts()) >= 5]
    df = df[df.language != 'Jupyter Notebook']
    clean_tokens = (df[column].apply(clean_html)
                              .apply(basic_clean)
                              .apply(tokenize)
                              .apply(remove_stopwords)
                   )
    
    for token in clean_tokens:
        token = ' '.join(token).split()
    
    df['stemmed'] = clean_tokens.apply(stem)
    df['lemmatized'] = clean_tokens.apply(lemmatize)
    return df

def wrangle_data():
    data = pd.read_json('data.json')
    return prepare_readme_data(data, 'readme_contents')