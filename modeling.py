import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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
    clean_tokens = (df[column].apply(clean_html)
                              .apply(basic_clean)
                              .apply(tokenize)
                              .apply(remove_stopwords)
                   )
    
    for token in clean_tokens:
        token = ' '.join(token).split()
    
    df['stemmed'] = clean_tokens.apply(stem)
    df['lemmatized'] = clean_tokens.apply(lemmatize)
    df['total_words'] = df['lemmatized'].str.split().str.len()
    df = df[df.total_words > 9]
    return df

# def wrangle_data():
#     data = pd.read_json('data.json')
#     return prepare_readme_data(data, 'readme_contents')

def wrangle_data():
    data = pd.read_csv('new_data.csv')
    return prepare_readme_data(data, 'readme_contents')

def split_df(df):
    '''
    This function performs split on repo data, stratifying on languages.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.language)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.language)
    return train, validate, test

def x_variables(train, validate, test):
    # Setup the X variables
    X_train = train.lemmatized
    X_validate = validate.lemmatized
    X_test = test.lemmatized
    return X_train, X_validate,X_test

def y_variables(train, validate, test):
    # Setup the y variables

    y_train = train.language
    y_validate = validate.language
    y_test = test.language
    return y_train, y_validate,y_test

# Create the tfidf vectorizer object
# encodes these values for classification purposes
def tfidf_object(X_train,X_validate,X_test):
    tfidf = TfidfVectorizer()

    # Fit on the training data
    tfidf.fit(X_train)
    # Use the object
    X_train_vectorized = tfidf.transform(X_train)
    X_validate_vectorized = tfidf.transform(X_validate)
    X_test_vectorized = tfidf.transform(X_test)
    return X_train_vectorized, X_validate_vectorized,X_test_vectorized

# Create the tfidf vectorizer object
# encodes these values for classification purposes
def count_vec_object(X_train,X_validate,X_test):
    cv = CountVectorizer(ngram_range=(1,3))
    # Fit on the training data
    cv.fit(X_train)
    # Use the object
    # Use the object
    X_train_cv = cv.transform(X_train)
    X_validate_cv = cv.transform(X_validate)
    X_test_cv = cv.transform(X_test)
    return X_train_cv, X_validate_cv,X_test_cv

def df_append(y_train, y_validate, y_test): 
    #creating a data frame that will be used to append any values
    train_app = pd.DataFrame(dict(actual=y_train))
    validate_app = pd.DataFrame(dict(actual=y_validate))
    test_app = pd.DataFrame(dict(actual=y_test))
    return train_app , validate_app , test_app 

#modeling - Logistic Regression (TFIDF Vectorizer)
def log_reg_tfidf_model(X_train_vectorized,y_train,X_validate_vectorized, y_validate,train_app, validate_app ):
    lm = LogisticRegression()
    lm.fit(X_train_vectorized, y_train)
    # Use the trained model to predict y given those vectorized inputs of X
    train_app['predicted_lm'] = lm.predict(X_train_vectorized)
    validate_app["predicted_lm"] = lm.predict(X_validate_vectorized)
    # test_app['predicted'] = lm.predict(X_test_vectorized)
    print(f'Train Accuracy Score: {lm.score(X_train_vectorized, y_train) * 100:.2f}%')
    print(f'Validate Accuracy Score: {lm.score(X_validate_vectorized, y_validate) * 100:.2f}%')

# modeling Decision tree with tfidf vectorizer 
def decision_tree_classifier_tfidf(X_train_vectorized,y_train,X_validate_vectorized, y_validate,train_app, validate_app):
    #decision tree classifier
    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    tree.fit(X_train_vectorized, y_train)
    # Use the trained model to predict y given those vectorized inputs of X
    train_app['predicted_t'] = tree.predict(X_train_vectorized)
    validate_app["predicted_t"] = tree.predict(X_validate_vectorized)
    # test_app['predicted'] = lm.predict(X_test_vectorized)
    print(f'Train Accuracy Score: {tree.score(X_train_vectorized, y_train) * 100:.2f}%')
    print(f'Validate Accuracy Score: {tree.score(X_validate_vectorized, y_validate) * 100:.2f}%')



def log_reg_count_vec(X_train_cv,y_train,X_validate_cv, y_validate,train_app, validate_app):
    # Use the trained model to predict y given those vectorized inputs of X
    lm = LogisticRegression()
    # Fit the classification model to vectorized train data
    lm.fit(X_train_cv, y_train)
    train_app['predicted_cv'] = lm.predict(X_train_cv)
    validate_app["predicted_cv"] = lm.predict(X_validate_cv)
    # test_app['predicted'] = lm.predict(X_test_vectorized)
    print(f'Train Accuracy Score: {lm.score(X_train_cv, y_train) * 100:.2f}%')
    print(f'Validate Accuracy Score: {lm.score(X_validate_cv, y_validate) * 100:.2f}%')

    # Train set Accuracy Matrix log regression on tfidf vectorizer
def log_reg_tfidf_acc_matrix(train_app):
    # accuracy matrix
    print('Accuracy: {:.2%}'.format(accuracy_score(train_app.actual, train_app.predicted_lm)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(train_app.predicted_lm, train_app.actual))
    print('---')
    print(classification_report(train_app.actual, train_app.predicted_lm))

def log_reg_tfidf_acc_val_matrix(validate_app):
    # accuracy matrix
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_app.actual, validate_app.predicted_lm)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(validate_app.predicted_lm, validate_app.actual))
    print('---')
    print(classification_report(validate_app.actual, validate_app.predicted_lm))

def acc_mat_decision_tree_tfidf(train_app):
    ## accuracy matrix Decision Tree Classifier (tfidf vectorizer)
    print('Accuracy: {:.2%}'.format(accuracy_score(train_app.actual, train_app.predicted_t)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(train_app.predicted_t, train_app.actual))
    print('---')
    print(classification_report(train_app.actual, train_app.predicted_t))

def acc_mat_decision_tree_val_tfidf(validate_app):
    ## accuracy matrix Decision Tree Classifier (tfidf vectorizer)
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_app.actual, validate_app.predicted_t)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(validate_app.predicted_t, validate_app.actual))
    print('---')
    print(classification_report(validate_app.actual, validate_app.predicted_t))

def cv_log_reg_matrix(train_app):
        ## accuracy matrix log regression Classifier (count vectorizer)
    print('Train Accuracy: {:.2%}'.format(accuracy_score(train_app.actual, train_app.predicted_cv)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(train_app.predicted_cv, train_app.actual))
    print('---')
    print(classification_report(train_app.actual, train_app.predicted_cv))


def cv_log_reg_matrix_val(validate_app):
        ## accuracy matrix log regression Classifier (count vectorizer)
    print('Train Accuracy: {:.2%}'.format(accuracy_score(validate_app.actual, validate_app.predicted_cv)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(validate_app.predicted_cv, validate_app.actual))
    print('---')
    print(classification_report(validate_app.actual, validate_app.predicted_cv))


# modeling Decision tree with tfidf vectorizer 
def decision_tree_classifier_cv_(X_train_cv,y_train,X_validate_cv, y_validate,X_test_cv,y_test, train_app, validate_app,test_app):
    #decision tree classifier
    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    tree.fit(X_train_cv, y_train)
    # Use the trained model to predict y given those vectorized inputs of X
    train_app['predicted_cv_dt'] = tree.predict(X_train_cv)
    validate_app["predicted_cv_dt"] = tree.predict(X_validate_cv)
    test_app['predicted_cv_dt'] = tree.predict(X_test_cv)
    print(f'Train Accuracy Score: {tree.score(X_train_cv, y_train) * 100:.2f}%')
    print(f'Validate Accuracy Score: {tree.score(X_validate_cv, y_validate) * 100:.2f}%')
    print(f'Test Accuracy Score: {tree.score(X_test_cv, y_test) * 100:.2f}%')

def cv_dec_tree_matrix(train_app):
    ## accuracy matrix decision tree Classifier (count vectorizer)
    print('Accuracy: {:.2%}'.format(accuracy_score(train_app.actual, train_app.predicted_cv_dt)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(train_app.predicted_cv_dt, train_app.actual))
    print('---')
    print(classification_report(train_app.actual, train_app.predicted_cv_dt))


def cv_dec_tree_matrix_val(validate_app):
    ## accuracy matrix decision tree Classifier (count vectorizer)
    print('Accuracy: {:.2%}'.format(accuracy_score(validate_app.actual, validate_app.predicted_cv_dt)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(validate_app.predicted_cv_dt, validate_app.actual))
    print('---')
    print(classification_report(validate_app.actual, validate_app.predicted_cv_dt))

def cv_dec_tree_matrix_test(test_app):
    ## accuracy matrix decision tree Classifier (count vectorizer)
    print('Accuracy: {:.2%}'.format(accuracy_score(test_app.actual, test_app.predicted_cv_dt)))
    print('---')
    print('Confusion Matrix')
    print(pd.crosstab(test_app.predicted_cv_dt, test_app.actual))
    print('---')
    print(classification_report(test_app.actual, test_app.predicted_cv_dt)) 