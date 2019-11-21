import re, functools, operator, collections, random
import numpy as np
import pandas as pd
from utils import *
from sklearn.utils import shuffle
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

def get_texts_bows_vs_years(corpus, dic, bigrams):
    bow_res = []
    for year in corpus:
        bows = []
        for doc in year:
            # Get tokens from documents:
            text = bigrams[simple_preprocess(doc)]
            # Convert tokens to BoW format:
            bow = dic.doc2bow(text)
            bows.append(bow)
        # Merge BoWs of 1 year into 1 big BoW:
        # result.append(merge_bows(bows, dic))
        bow_res.append(bows)
    return bow_res

def tokens_bows_dict(docs, no_below, no_above, min_count, threshold,
        bigrams=True):
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
    texts = [simple_preprocess(doc) for doc in docs]

    if bigrams:
        tmp = Phrases(texts, min_count=min_count, threshold=threshold)
        bigrams = Phraser(tmp)
        texts = [bigrams[doc] for doc in texts]

    # Create a dictionary from 'docs' containing
    # the number of times a word appears in the training set:
    dictionary = gensim.corpora.Dictionary(texts)

    # Filter extremes vocabularies:
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Create the Bag-of-words model for each document i.e for
    # each document we create a dictionary reporting how many
    # words and how many times those words appear:
    bows = [dictionary.doc2bow(text) for text in texts]

    return texts, bows, dictionary, bigrams

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

def models_codherence_perplexity(texts, bows, dic,
        topic_start=10, topic_end=201, step=10,
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
        model = LdaMulticore(corpus=bows, id2word=dic,
                             eta='auto', num_topics=num_topics,
                             chunksize=chunk, passes=passes, workers=cores)
        # Build coherence model to test the topic model:
        coherence_model = CoherenceModel(model=model, texts=texts,
                                         dictionary=dic, coherence='c_v')
        # Save the results:
        models.append(model)
        coherence_scores.append(coherence_model.get_coherence())
        perplexity_scores.append(model.log_perplexity(bows))
    return models, coherence_scores, perplexity_scores

def word_histogram(bows, model, dic):
    topics = [[topic[0] for topic in model[bow]] for bow in bows]
    topics = functools.reduce(operator.iconcat, topics, [])
    topic_map = dict(model.show_topics(model.num_topics))
    topic_map = {k: re.findall(r'[a-z]+', v) for (k, v) in topic_map.items()}
    words = [topic_map[topic] for topic in topics]
    words = functools.reduce(operator.iconcat, words, [])
    return [(dic[p[0]], p[1]) for p in dic.doc2bow(words)]

def topic_union(top_topics, topic_list, corr, num):
    """ Get a collection of preference topics.
        Preference topics is consist of top topics w.r.t
        coherence score, union with top topics that are least
        correlated with other topics.
    """
    # Get the topic map:
    topic_map = dict(topic_list)
    topic_map = {k: re.findall(r'[a-z_]+', v) for k, v in topic_map.items()}
    topic_map = {''.join(v): k for k, v in topic_map.items()}
    # Get the top independent topics:
    corr_sum = np.sum(corr, axis=1)
    top_independence = []
    for _ in range(num):
        top_index = np.argmax(corr_sum)
        corr_sum[top_index] = 0
        top_independence.append(top_index)
    # Get the top coherence topics:
    top_coherence = [[q[1] for q in p[0]] for p in top_topics[:num]]
    top_coherence = [''.join(presentation) for presentation in top_coherence]
    top_coherence = [topic_map[pre] for pre in top_coherence]
    return sorted(list(set(top_independence).union(set(top_coherence))))

def convert_topic(topics, union, corr):
    for i in range(len(topics)):
        if topics[i] not in union:
            corr_score = corr[topics[i]][union]
            topics[i] = union[np.argmin(corr_score)]
    return topics

def topic_histogram(bows, model, min_prob, union, corr):
    tmp = [model.get_document_topics(bow, minimum_probability=min_prob)
            for bow in bows]
    tmp = [[p[0] for p in l] for l in tmp]
    topics = [convert_topic(topics, union, corr) for topics in tmp]
    topics = functools.reduce(operator.iconcat, topics, [])
    hist = {topic: 0 for topic in union}
    for topic in topics:
        hist[topic] += 1
    hist = list(hist.items())
    hist.sort()
    return hist

def topic_hist_years(corpus, model, min_prob, union, corr):
    return [topic_histogram(bows, model, min_prob, union, corr)
            for bows in corpus]

def sampling_corpus(corpus, percent=0.2):
    sample = []
    num = int(len(corpus[0])*percent)
    for cor in corpus:
        sample = sample + random.choices(cor, k=num)
    return sample

def run(office, sector, start_year, end_year):
    """ Analyze topics for the office and sector.

    """
    # Get the training documents from 1 year:
    corpus = query_intersection(2010, 2019, office, sector, False)
    # Sampling documents in each year for training:
    docs = sampling_corpus(corpus, percent=1/(end_year - start_year))
    # Covert documents to tokens, bag of word and dictionary format:
    texts, bows, dic, bigrams = tokens_bows_dict(docs, 2, 0.5, 2, 80, True)
    # Build models for comparison:
    start = max(len(docs) - 70, 10)
    end = len(docs) + 1
    step = 10
    models, coherences, perplexities = models_codherence_perplexity(
            texts, bows, dic,                            \
            topic_start=start, topic_end=end, step=step, \
            chunk=20, passes=3)
    # Choose a good model:
    per = [-p for p in perplexities]
    per = [(p - min(per))/(max(per) - min(per)) for p in per]
    which = [per[i]*coherences[i] for i in range(len(per))]
    which = np.argmax(which)
    chosen = models[which]
    # Get texts and bows for each year:
    bows_vs_years = get_texts_bows_vs_years(corpus, dic, bigrams)
    # Prepare to get topic union:
    topic_list = chosen.show_topics(chosen.num_topics, 10)
    top_topics = chosen.top_topics(texts=texts, coherence='c_v', topn=10)
    # Get the correlation matrix:
    mdiff, _ = chosen.diff(chosen, distance='jaccard', num_words=100)
    # Get top topics based on coherence and correlation:
    union = topic_union(top_topics, topic_list, mdiff, 10)
    # Get the count for each topic in each year:
    hists = topic_hist_years(bows_vs_years, chosen, 0.01, union, mdiff)
    # Get DataFrame:
    data = [[p[1] for p in hist] for hist in hists]
    pre = [' | '.join(re.findall(r'[a-z_]+', topic_list[i][1])) \
            for i in union]
    df = pd.DataFrame(data, columns=pre, index=range(2010, 2019))
    # Save the model:
    df.to_csv(os.getcwd()[:-14] + '/web/source/topicm'+"_"+office+"_"+sector+".csv")
