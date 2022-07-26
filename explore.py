#ignore warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import trigrams
from nltk import bigrams
from collections import Counter
from typing import Dict, List, Optional, Union, cast


import os
import time
import csv


import requests
import prepare

from sklearn.model_selection import train_test_split

from scipy import stats
from scipy.stats import pearsonr, spearmanr



ADDITIONAL = ['The', 'I', 'This', 'app', 'run', 'project', 'user', 'use', 'mental', 'file', 'health',
       'create', 'page', 'code', 'also', 'help', 'used']
             
def clean(readme_contents):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL
    text = (unicodedata.normalize('NFKD', readme_contents)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    #words = re.sub(r'[^a-z\s]', '', text).split()
    words = re.sub (r'([^a-zA-Z ]+?)', "", text).split()
    #words = re.sub(r'[\D]', '', words).split()
    
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def made_dfs(df):
#making dfs with clean function for all langs
    JavaScript_words = clean(' '.join(df[df.language == 'JavaScript'].readme_contents))
    HTML_words = clean(' '.join(df[df.language == 'HTML'].readme_contents))
    Python_words = clean(' '.join(df[df.language == 'Python'].readme_contents))
    Java_words = clean(' '.join(df[df.language == 'Java'].readme_contents))
    R_words = clean(' '.join(df[df.language == 'R'].readme_contents))
    all_words = clean(' '.join(df.readme_contents))

    return JavaScript_words, HTML_words, Python_words, Java_words, R_words, all_words


#make dfs to show frequencies of words

def make_df_freqs(JavaScript_words, HTML_words, Python_words, Java_words, R_words, all_words):

    JavaScript_freq = pd.Series(JavaScript_words).value_counts()
    HTML_freq = pd.Series(HTML_words).value_counts()
    Python_freq = pd.Series(Python_words).value_counts()
    Java_freq = pd.Series(Java_words).value_counts()
    R_freq = pd.Series(R_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()


    word_counts = (pd.concat([all_freq, JavaScript_freq, HTML_freq, Python_freq, 
                    Java_freq, R_freq], axis=1, sort=True)
                .set_axis(['all_words', 'JavaScript', 'HTML', 'Python', 'Java', 'R'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))

    return word_counts

def plot_bins(df):
    #plt.figure(figsize=(10,8))
    conditions = [(df.total_words > 171),
    (df.total_words >= 51) & (df.total_words <= 171),
              (df.total_words < 51)]
    choices = ['high_count', 'med_count', 'low_count']
    df['count_bin'] = np.select(conditions, choices)
    df_plot = df.groupby(['language', 'count_bin']).size().reset_index().pivot(columns='count_bin', index='language', values=0)
    df_plot.plot(kind='bar',
                 figsize = (10, 8))
    plt.xlabel('Language')
    plt.title('Programming Languages with Counts Bins')

def stats_test_1(word_counts):
    α = 0.05
    js_sample = word_counts.JavaScript
    overall_mean = word_counts.all_words.mean()
    t, p = stats.ttest_1samp(js_sample, overall_mean)
    print(f'P Value: {p/2:.3f}')
    if p/2 < α and t > 0:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis.')

def stats_test_2(word_counts):

    #test for equal variance 
    #H0 is that the variances are equal
    #Ha is that the variances are not equal
    # if p > .05, variances are not significantly different and set argument to equal_var = True
    #if p < .05, variances are significantly different and set argument to equal_var = False
    #Levene test on two groups

    #set alpha
    α = 0.05

    #perform test to determine variance
    f, p = stats.levene(word_counts.JavaScript,
                word_counts.Python)

    #evaluate coefficient and p-value
    print(f'Levene\'s F-statistic: {f:.3f}\nP-value: {p:.3f}')

    #evaluate if 
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def stats_test_2_t(word_counts):
    #set alpha
    α = 0.05

    #perform test
    t, p = stats.ttest_ind(word_counts.JavaScript, word_counts.Python, equal_var = False)

    #print p-value
    print(f'P Value: {p/2:.3f}')

    #evaluate if mean of the word counts associated w/ JavaScript is significantly higher than the mean 
    # of the mean word counts associated with Python, is p/2 < a and t > 0?
    if p/2 < α and t > 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def stats_test_3(word_counts):

    #test for equal variance 
    #H0 is that the variances are equal
    #Ha is that the variances are not equal
    # if p > .05, variances are not significantly different and set argument to equal_var = True
    #if p < .05, variances are significantly different and set argument to equal_var = False
    #Levene test on two groups
    #set alpha
    α = 0.05

    #perform test to determine variance
    f, p = stats.levene(word_counts.R,
                word_counts.Python)
    #evaluate coefficient and p-value
    print(f'Levene\'s F-statistic: {f:.3f}\nP-value: {p:.3f}')

    #evaluate if 
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def stats_test_4(word_counts):
    #set alpha
    α = 0.05

    #perform test
    t, p = stats.ttest_ind(word_counts.R, word_counts.Python, equal_var = False)

    #print p-value
    print(f'P Value: {p/2:.3f}')

    #evaluate if mean of the word counts associated w/ R is significantly lower than the mean 
    # of the mean word counts associated with Python, is p/2 < a and t < 0?
    if p/2 < α and t < 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def top_words(word_counts):

    #makes dfs for arranging top words by the diff langs to help compare
    #if tops in some langs may be way different than tops in others
    js_top = word_counts.sort_values(by='JavaScript', ascending=False).head(20)
    html_top = word_counts.sort_values(by='HTML', ascending=False).head(20)
    python_top = word_counts.sort_values(by='Python', ascending=False).head(20)
    java_top = word_counts.sort_values(by='Java', ascending=False).head(20)
    r_top = word_counts.sort_values(by='R', ascending=False).head(20)

    top_words = (pd.concat([js_top, html_top, python_top, java_top,
                        r_top], axis=0, sort=True)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))


    top_words= top_words.drop(columns=['all_words'])
    tops = top_words[top_words.index.value_counts() > 1]
    return tops.index.unique()

def plot_word_perc(word_counts):
#making the new df with top counts
    most_freq = (word_counts[word_counts['all_words'] > 730])
    #changing df to show percentages and transposing to visualize best
    word_perc = pd.DataFrame()
    for col in most_freq.columns:
        word_perc = word_perc.append(most_freq[col].sort_values(ascending=False)/most_freq[col].sum())

    word_perc_T = word_perc.T
    word_perc_T = word_perc_T
    word_perc_T = word_perc_T.drop(columns=['all_words'])
    #dropping the all column prior to graphing just bc it's not relevant to
    #the question and don't want to cloud visual
    word_perc_T
    plt.rcParams["figure.figsize"] = (10,8)
    word_perc_T.plot(kind = 'barh', stacked=True)

    plt.title('Proportion of Languages for the 20 Most Common Words')
    plt.show()


def plot_bigrams(all_words, JavaScript_words, HTML_words, Python_words, Java_words, R_words):
    #this shows the 20 top most frequently occuring birgrams

    bigrams_all = (pd.Series(nltk.ngrams(all_words, 2))
                        .value_counts())
    bigrams_JavaScript = (pd.Series(nltk.ngrams(JavaScript_words, 2))
                        .value_counts())
    bigrams_HTML = (pd.Series(nltk.ngrams(HTML_words, 2))
                        .value_counts())
    bigrams_Python = (pd.Series(nltk.ngrams(Python_words, 2))
                        .value_counts())
    bigrams_Java = (pd.Series(nltk.ngrams(Java_words, 2))
                        .value_counts())
    bigrams_R = (pd.Series(nltk.ngrams(R_words, 2))
                        .value_counts())

    bigram_counts = (pd.concat([bigrams_all, bigrams_JavaScript, bigrams_HTML, bigrams_Python, 
                        bigrams_Java, bigrams_R], axis=1, sort=True)
                    .set_axis(['all', 'JavaScript', 'HTML', 'Python', 'Java', 'R'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    most_freq_bigrams = (bigram_counts[bigram_counts['all'] >= 105])

    word_perc_bigrams = pd.DataFrame()
    for col in most_freq_bigrams.columns:
        word_perc_bigrams = word_perc_bigrams.append(most_freq_bigrams[col].sort_values(ascending=False)/most_freq_bigrams[col].sum())

    word_perc_bigrams_T = word_perc_bigrams.T
    word_perc_bigrams_T = word_perc_bigrams_T.drop(columns=['all'])
    #dropping the all column prior to graphing just bc it's not relevant to
    #the question and don't want to cloud visual

    # plt.rcParams["figure.figsize"] = (10,8)
    # word_perc_bigrams_T.plot(kind = 'barh', stacked=True)
    # plt.title('Proportion of languages for the 20 most common bigrams')

    return word_perc_bigrams_T

def plot_bigrams_10(word_perc_bigrams_T):
    #This shows bigrams that have higher distribution (above 10%) in a certain languages
    #this can be used to show what bigrams may be most predictive regardless of
    #overall frequency
    Java_high = word_perc_bigrams_T[word_perc_bigrams_T.Java > .1]
    Python_high = word_perc_bigrams_T[word_perc_bigrams_T.Python > .1]
    R_high = word_perc_bigrams_T[word_perc_bigrams_T.R > .1]
    HTML_high = word_perc_bigrams_T[word_perc_bigrams_T.HTML > .1]
    js_high = word_perc_bigrams_T[word_perc_bigrams_T.JavaScript > .1]

    langs_high_tris = pd.concat([HTML_high, R_high, Python_high, Java_high, js_high])

    plt.rcParams["figure.figsize"] = (10,8)
    word_perc_bigrams_T.plot(kind = 'barh', stacked=True)
    plt.title('Proportion of Languages for the 20 Most Common Bigrams', size=20)
    plt.rcParams["figure.figsize"] = (10,8)
    langs_high_tris.plot(kind = 'barh', stacked=True)
    plt.title('High Frequency Bigrams of Languages Compared', size= 20)

def plot_trigrams(all_words, HTML_words, Python_words, Java_words, R_words):
    #This shows trigrams that have higher distribution (above 5%) in a certain languages
    #this can be used to show what trigrams may be most predictive regardless of
    #overall frequency
    #note: high frequency trigrams are very evenly distributed across
    #languages therefore the high frequency trigrams are not shown as there 
    #is no potential predictivity there 

    trigrams_all = (pd.Series(nltk.ngrams(all_words, 3))
                        .value_counts())
    trigrams_HTML = (pd.Series(nltk.ngrams(HTML_words, 3))
                        .value_counts())
    trigrams_Python = (pd.Series(nltk.ngrams(Python_words, 3))
                        .value_counts())
    trigrams_Java = (pd.Series(nltk.ngrams(Java_words, 3))
                        .value_counts())
    trigrams_R = (pd.Series(nltk.ngrams(R_words, 3))
                        .value_counts())

    trigram_counts = (pd.concat([trigrams_all, trigrams_HTML, trigrams_Python, 
                        trigrams_Java, trigrams_R], axis=1, sort=True)
                    .set_axis(['all', 'HTML', 'Python', 'Java', 'R'], axis=1, inplace=False)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    most_freq_trigrams = (trigram_counts[trigram_counts['all'] >= 30 ])
    most_freq_trigrams = (most_freq_trigrams[most_freq_trigrams['all'] < 60])

    word_perc_trigrams = pd.DataFrame()
    for col in most_freq_trigrams.columns:
        word_perc_trigrams = word_perc_trigrams.append(most_freq_trigrams[col].sort_values(ascending=False)/most_freq_trigrams[col].sum())

    word_perc_trigrams_T = word_perc_trigrams.T
    word_perc_trigrams_T = word_perc_trigrams_T.drop(columns=['all'])
    #dropping the all column prior to graphing just bc it's not relevant to
    #the question and don't want to cloud visual

    Java_high = word_perc_trigrams_T[word_perc_trigrams_T.Java > .05]
    Python_high = word_perc_trigrams_T[word_perc_trigrams_T.Python > .05]
    R_high = word_perc_trigrams_T[word_perc_trigrams_T.R > .05]
    HTML_high = word_perc_trigrams_T[word_perc_trigrams_T.HTML > .05]

    langs_high_tris = pd.concat([HTML_high, R_high, Python_high, Java_high])

    plt.rcParams["figure.figsize"] = (15,3)
    langs_high_tris.plot(kind = 'barh', stacked=True)
    plt.title('High Frequency Trigrams of Languages Compared', size=20)

def tri_test_1():
    alpha = .05
    index = ['all', 'HTML']
    columns = ['total_words', 'trigram']

    observed = pd.DataFrame([[302457, 50], [60208, 35]], index=index, columns=columns)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def tri_test_2():
    alpha =.05
    index = ['all', 'Python']
    columns = ['total_words', 'trigram']

    observed = pd.DataFrame([[302457, 38], [57710, 21]], index=index, columns=columns)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def tri_test_3():
    alpha = .05
    index = ['all', 'Python']
    columns = ['total_words', 'trigram']

    observed = pd.DataFrame([[302457, 41], [57710, 25]], index=index, columns=columns)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def tri_test_4():
    alpha=.05
    index = ['all', 'HTML']
    columns = ['total_words', 'trigram']

    observed = pd.DataFrame([[302457, 58], [60208, 3]], index=index, columns=columns)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')


def split_data(df):
    '''
    This function takes in a dataframe and splits it into three subgroups: train, test, validate
    for proper evalution, statistical testing, and modeling. Three dataframes are returned.
    '''
    #train, test, split
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.language)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.language)
    return train, validate, test

def establish_baseline():
    df = prepare.wrangle_data()
    langs = pd.concat([df.language.value_counts(),
                    df.language.value_counts(normalize=True)], axis=1)
    langs
    langs.columns = ['counts', 'percent']
    #establish baseline
    baseline = langs.loc['JavaScript', 'percent']

    print(f'Baseline is: ',  f'{baseline * 100:.2f}%')
    
def visuals():
    df = prepare.wrangle_data()
    sns.catplot(data = df,
           x = 'total_words',
           y = 'language',
           kind = 'violin',
           height = 8,
           aspect = 10/8)
    sns.despine(left = True, bottom = True)   
    plt.ylabel('Language')
    plt.xlabel('Total Words')
    plt.title('Word Counts per Programming Language', size=20)
    
   