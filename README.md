# NLP Project - Mental Health Repositories

#### Project Collaborators (with link to GitHub):
- Jules Morris - https://github.com/JulesMorris
- Lincoln Muriithi - https://github.com/lincolnmuriithi11
- Mindy Shiben - https://github.com/mindyshiben

## Project Description
Our team designed and executed a Natural Language Processing(NLP) project which anaylzes GitHub readme files on the topic of mental health.
The goal of this project is to use Natural Language Processing in order to build a predictive classification model of programming languages using the text from mental health Github repository's readme.md. 
The five programming languages in data acquired for this project are: JavaScript, Python, HTML, Java, and R which are the most common languages in GitHub repositories on the subject matter of mental health.
Therefore, the predictive model aims to predict the programming languages of repositories that were written in one of these five languages.

## Link to presentation slides:

https://www.canva.com/design/DAFHK3Si990/kUSxdA09Nt2x506IJYaylA/edit?utm_content=DAFHK3Si990&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Executive Summary
For this project our team chose to focus on the topic of mental health, and formed an initial hypothesis that due to the academic quantitative analysis of mental health, that certain languages, such as R, would be robust in the scrapped data.
After some initial exploration of Github, we chose Python, Java, R, HTML, and JavaScript as the only languages of focus to build the predictive model. After acquisition and preparation, we utilized 1627 records in data exploration and modeling. The Baseline language is JavaScript which accounted for 37% of our corpus
The models used were a TFIDF and Count Vectorizer Logistic Regression and a TFIDF and Count Vectorizer Decision Tree model. The best performing model is the Count Vectorized Decision Tree which has slightly over 50% accuracy (improvement of 48% from baseline).

## Initial Thoughts
- Do certain languages have a higher average readme word count than others?
- Are certain frequently occuring words shared across more than one language?
- What words, if any, are frequently occuring in readmes for certain langauges that are barely prevelant in other language readmes?
- Can small groups of sequential words (bigrams and trigrams) in readmes be predictive of the target variable?

## Data Dictionary

- `language` : programming language (target variable)
- `readme_contents` : raw readme text   
- `stemmed` : readme_contents cleaned + Porter stemmer  
- `lemmatized` : readme_contents cleaned + WordNetLemmatizer()
- `total_words` : number of words in readme text
- `langs ` : counts and distributions of programming languages 

## Project Plan:

- Acquire data by webscraping Github to obtain text from the readmes of repositories on the topic of mental health in the five target programming languages.

- Data sources are as follows:

>- https://github.com/search?l=JavaScript&q=mental+health&type=Repositories
>- https://github.com/search?l=HTML&q=mental+health&type=Repositories
>- https://github.com/search?l=Python&q=mental+health&type=Repositories
>- https://github.com/search?l=Java&q=mental+health&type=Repositories
>- https://github.com/search?l=R&q=mental+health&type=Repositories

- Save data as .csv files
- Use the pandas library to read the .csv files and concatenate them into one dataframe
- Clean and prepare the data, create acquire.py and prepare.py to automate automate processes.
- Univariate exploration of the and intial thoughts/questions to create a framework for the exploratory process.
- Explore hypotheses by visualizing relationships and conducting statistical tests.
- Document findings of the exploratory process.
- Discovery baseline accuracy for modeling using the target variable's mode.
- Split data into train, validate and test subsets.
- Create and train predictive classification models and analyze model performance
- Choose the model with that performs the best and evaluate using test data.
- Construct Final Report & slides for presentation
- Create README.md which details project goals, process, findings, and replication steps

## Conclusion

Our team utilized the data science pipeline and successfully answered initial questions as well as built an improved predictive model.

Through out exploration, we discovered the following:

Do certain languages have a higher average readme word count than others?
>-  On average, Java readmes have the lowest total words whereas JavaScript has the highest. JavaScript readmes have considerably more words than the other 4 programing languages. 

Are certain frequently occuring words share across more than one language?
>- The following words are amongst the top 20 frequently occuring words (excluding stop words) in 2 or more languages:
'section', 'test', 'install', 'make', 'need', 'feature', 'user', 'time',
'information', 'file', 'command', 'people', 'one', 'application','model', 'dataset', 'analysis', 'following'

What words, if any, are frequently occuring in readmes for certain langauges that are barely prevelant in other language readmes?
>- Looking at graphs (and stat tests), these top occuring words may be helpful in identifying said languages:
    - 'data': R
    - 'file': R
    - 'website': HTML
    - 'application': Java
    - 'react', 'build', 'npm': JavaScript

Can small groups of sequential words (bigrams and trigrams) in readmes be predictive of the target variable?
>- Many of the most indentifying bigrams are shared with 2 or more languages, yet the 5 trigrams that showed the most predictive potential of the target which has been confirmed with chi squared testing. Four out of the five languages seem to have a predictive trigram (a useful JavaScript trigram was not seen).

Through modeling, we accomplished the following:

Our was able to build, fit, and train a Count Vectorizer Decision Tree model which is able to predict the programming language of GitHub readme files on the topic of mental health with ~ 50% accuracy.
With a 48% improvement from baseline seen on all 3 subsets (train, validate, and test), we are confident that this model's accuracy is represented correctly and we can recommend utilization of this model at this time.
However, we do believe that with more time and resources, we could improve this model's accuracy.

Our next steps include the following:

- Acquiring more data on this subject matter
- Further exploring readme contents 
- Feature engineer additional variables


## Reproduce this project
Clone this repository. Acquire the data by requesting github token access and using acquire.py. Prior to running the
acquire.py script, your credentials should be saved on your device in a .env file in addition to a .gitignore file which includes your
credentials. Scraping the data may take several hours. The acquire.py file will save the data at 5 different .json files (the data will be
merged in the next step). After acquiring the data, run the "final_notebook" in this repository. Standard data science libraries
(Pandas, Numpy, Statsmodels, SKLearn, Matplotlib, Seaborn, etc.) along with WordCloud, nltk, and Beautiful Soup, will be required to run the notebook
and may need to be installed on your device if they do not already exist.

