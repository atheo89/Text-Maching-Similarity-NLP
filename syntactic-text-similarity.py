import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_value_text(data):
    """
    Description:
        Extracts the values of each field of the json, but returns 
        only a unified string of the text_values after grouping.
    Input Parameters
        data: json data [contact or respond data] 
    Output Parameters
        _txt_str: String 
    """
    _txt = []
    _misc =[]
    for i in range(len(data)): 
        for key, value in data[i].items(): 
            if key == 'text_value':
                _txt.append(value)
            else:
                _misc.append(value)
    _txt_str=' '.join(_txt) 

    return _txt_str

def string_constractor(data, kind):
    """
    Description:
        Composes the unified strings into lists and gives
        a tag according to the kind of data that red
    Input Parameters
        data: json data [contact or respond data] 
        kind: String 
    Output Parameters
        text: List
        tag: List 
    """
    text = []
    tag = []
    for i in range(len(data)):
        text.append(extract_value_text(data[i]))
        tag.append(kind + str(i))
    return text , tag

def df_constructor(text_c, tag_c, text_r, tag_r):
    """
    Description:
        Composes a dataframe from lists 
    Input Parameters
        text_c: List
        tag_c: List 
        text_r: List
        tag_r: List
    Output Parameters
        df: Dataframe
    """
    text = []
    tag = []
    text = text_c + text_r
    tag = tag_c + tag_r

    df = pd.DataFrame({'text': text, 'tag': tag })
    return df

def heatmap(df, mtx):
    """
    Description:
        Visualizes the cosine similarity between the inputs 
    Input Parameters
        df: Dataframe
        mtx: 2D Array 
    """
    x_labels = df['text'].tolist()
    y_labels = df['text'].tolist()

    fig, ax = plt.subplots()
    im = ax.imshow(mtx)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, '%.2f'%mtx[i, j], ha='center', va='center', color='w', fontsize=6)
    plt.show()

def preprocess_text(doc):
    """
    Description:
        Preprocesses each document into a proper 
        format to to pass it on TfidfVectorizer
    Input Parameters
        doc: String
    """
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(doc)
    
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords

def feature_matrix_tf_idf(df):
    """
    Description:
        Preprocess texts and Constracts a feature matrix
    Input Parameters
        df: Dataframe
    Output Parameters
        X_ftr: Compressed sparse row matrix 
    """
    corpus = df['text'].tolist()

    vectoriser = TfidfVectorizer(analyzer=preprocess_text)
    X_ftr = vectoriser.fit_transform(corpus)

    return X_ftr

def pairwise_cosine_similarity(X_ftr):
    """
    Description:
        Measures the similarity between two non-zero 
        vectors of an inner product space
    Input Parameters
        X_ftr: Compressed sparse row matrix 
    Output Parameters
        res: 2d Array
    """
    pairwise_similarity = X_ftr * X_ftr.T
    res = pairwise_similarity.toarray()

    return res
    
def query_res(corr, df):
    """
    Description:
        Finds the index of the most similar document 
    Input Parameters
        corr: 2d Array
        df: Dataframe
    Output Parameters
        qdf: Dataframe
    """
    corpus = df['text'].tolist()

    np.fill_diagonal(corr, np.nan)
    flt = df[df['tag'].str.contains('contact',regex = False)]
    n = len(flt)
    lst = []
    for idx in range(len(flt)):

        query_idx = corpus.index(flt['text'][idx])       
        result_idx = np.nanargmax(corr[query_idx])
        score = corr.item(query_idx,result_idx)

        lst.append([df['tag'][query_idx],df['tag'][result_idx],score ])
        res = pd.DataFrame(lst, columns =['contact', 'response', 'score'])    
        qdf = res[res['response'].str.contains('respond ',regex = False)]
 

    return qdf


if __name__ == '__main__':

    try:
        print(' - [ ? ] Loading test data')
        with open('data.json') as json_file: 
            data = json.load(json_file)
        contact =[]
        respond = []
        contact = data['contact']
        respond = data['respond']
        print('   [ V ] Successfully complete')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')
        
    try:
        print(' - [ ? ] Preparing the text_value into a uniform string')
        text_c, tag_c = string_constractor(contact, 'contact ')
        text_r, tag_r = string_constractor(respond, 'respond ')
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')
        
    try:
        print(' - [ ? ] Constructing a dataframe')
        df = df_constructor(text_c, tag_c, text_r, tag_r)
        print(df)
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')

    try:
        print(' - [ ? ] Creating a feature matrix of words\'s frequencies')
        X_train = feature_matrix_tf_idf(df)
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')

    try:
        print(' - [ ? ] Calculating the cosine similarity array')
        corr = pairwise_cosine_similarity(X_train) 
        print(corr)
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')

    try:
        print(' - [ ? ] Vissualizing the similarity array')
        heatmap(df, corr)
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')

    try:
        print(' - [ ? ] Matching the contact/respond segments with the max similarity')
        print(query_res(corr, df))
        print('   [ V ] Successfully complete')
        print('')
    except Exception as e:
        print(e)
        print('   [ X ] Failed. Application exits...')
