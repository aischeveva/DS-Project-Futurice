import gensim
import pandas as pd
from utils import *
from sklearn.utils import shuffle
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


def tokens_bows_dict(docs, no_below, no_above):
    """ Get the bag-of-word form and dictionary from the document corpus.
        --------------------
        Parameter:
            docs: document corpus
            no_below: filter words that appear in less than
                      'no_below' number of document.
            no_above: filter words that appear in more than
                      'no_above' percent of document.

        Return:
            (bow corpus, dictionary)
    """
    # Tokenize documents:
    tokenized_docs = [gensim.utils.simple_preprocess(doc) for doc in docs]

    # Create a dictionary from 'docs' containing
    # the number of times a word appears in the training set:
    dictionary = gensim.corpora.Dictionary(tokenized_docs)

    # Filter extremes vocabularies:
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Create the Bag-of-words model for each document i.e for
    # each document we create a dictionary reporting how many
    # words and how many times those words appear:
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    return tokenized_docs, bow_corpus, dictionary

def train_test_split(texts, bows, test_size=0.2):
    """ Split corpus into train and test set.
        --------------------
        Parameter:
            texts (list of list of string): tokenized documents
            bows: list of documents in 'Bag of Words' format
            test_size: percentage of test documents

        Return:
            texts_train, texts_test, bows_train, bows_test
    """
    split = int(len(texts) * (1 - test_size))
    texts_shuf, bows_shuf = shuffle(texts, bows)
    texts_train, texts_test = texts_shuf[:split], texts_shuf[split:]
    bows_train, bows_test = bows_shuf[:split], bows_shuf[split:]
    return texts_train, texts_test, bows_train, bows_test

def models_codherence_perplexity(texts_train, texts_test, bows_train,
        bows_test, dic, topic_start=10, topic_end=201, step=10,
        chunk=10, passes=3, cores=2):
    """ Build models on a range of number of topics to compare quality.
        The output is 3 lists of:
            1. List of built models
            2. List of coherence scores calculated on training data
            3. List of perplexity scores calculated on test data
        --------------------
        Parameter:
            texts_train, texts_test, bows_train, bows_test
            dic: dictionary of id <-> word
            topic_start, topic_end, step: range of number of topics
            chunk: number of data used in each training step
            passes: number of passes through the whole training data
            cores: number of cores use for parallel training

        Return:
            models, coherence_scores, perplexity_scores
    """
    models = []
    coherence_scores = []
    perplexity_scores = []
    for num_topics in range(topic_start, topic_end, step):
        print('Building model of %d topics' % (num_topics))
        # Build topic model for the given number of topics:
        model = LdaMulticore(corpus=bows_train, id2word=dic,
                             eta='auto', num_topics=num_topics,
                             chunksize=chunk, passes=passes, workers=cores)
        # Build coherence model to test the topic model:
        coherence_model = CoherenceModel(model=model, texts=texts_train,
                                         dictionary=dic, coherence='c_v')
        # Save the results:
        models.append(model)
        coherence_scores.append(coherence_model.get_coherence())
        perplexity_scores.append(model.log_perplexity(bows_test))
    return models, coherence_scores, perplexity_scores

'''
def topic_modeling(num_topics, passes, num_cores, chunk, freq,
        train_year, office, sector, companies=['']):
    """ Use DA topic modeling to extract topic dictribution
        of documents over years.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            companies (list of str): list of interested companies
            num_topics: fixed number of topics to be extracted
            passes: number of passes through document data in training
            num_cores: number of processors used in training

        Return:
            (pd.DataFrame, lda_model)
    """
    # Query desired reports for tf-idf.
    docs = query_docs(train_year, train_year+1, office,
            sector, False, companies)

    # Get Bag-of-Words format for the docs.
    bow_corpus, dictionary = get_bow_corpus(docs[0], freq)

    # Train lda model using gensim.models.LdaMulticore and save it to 'lda_model'
    lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                            num_topics = num_topics,
                                            chunksize = chunk,
                                            id2word = dictionary,
                                            eta = 'auto',
                                            passes = passes,
                                            workers = num_cores)


    return lda_model, bow_corpus, dictionary
'''
