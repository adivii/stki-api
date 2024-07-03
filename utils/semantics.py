import gensim
import nltk
import json
import pandas as pd
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
from .clustering import get_cluster

dictionary = None
title_tfidf_model = None
title_lsi_model = None
title_index = None
df = None


def init(cluster_: int):
    nltk.download('punkt')
    load_data(cluster_)

    global dictionary
    global title_tfidf_model
    global title_lsi_model
    global title_index

    df['token'] = df['title_abstract_clean_stem_stop'].apply(lambda x: nltk.word_tokenize(x))
    dictionary = corpora.Dictionary(df['token'])
    corpus = [dictionary.doc2bow(title) for title in df['token']]

    print(df.shape)

    title_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
    title_lsi_model = gensim.models.LsiModel(title_tfidf_model[corpus], id2word=dictionary, num_topics=10)

    title_tfidf_corpus = title_tfidf_model[corpus]
    title_lsi_corpus = title_lsi_model[title_tfidf_corpus]

    title_index = MatrixSimilarity(title_lsi_corpus)

    return True


def load_data(cluster_):
    global df
    df = pd.read_csv('datas/Model_005.csv')
    df.dropna(subset=['title_abstract_clean_stem_stop'], inplace=True)
    df = df[df['cluster'] == cluster_]
    df.reset_index(drop=True, inplace=True)


def search_similar_titles(search_term):
    temp = get_cluster(search_term)
    cluster_ = temp['cluster']
    init(cluster_)

    query_bow = dictionary.doc2bow(nltk.word_tokenize(search_term))
    query_tfidf = title_tfidf_model[query_bow]
    query_lsi = title_lsi_model[query_tfidf]

    title_index.num_best = 10

    titles_list = title_index[query_lsi]

    titles_list.sort(key=itemgetter(1), reverse=True)
    title_names = []

    for j, title in enumerate(titles_list):

        author_list = []
        temps = df['creators'][title[0]].replace('\'', '\"')
        temps = temps.replace('None', '\"\"')
        authors = json.loads(temps)

        for author in authors:
            author_list.append(" ".join([author['name']['given'], author['name']['family']]))

        title_names.append(
            {
                'relevance': round((title[1] * 100), 2),
                'title': df['title'][title[0]].title(),
                'divisions': df['prodi'][title[0]],
                'author': author_list,
                'url': df['uri'][title[0]],
            }

        )
        if j == (title_index.num_best - 1):
            break

    return {"cluster": int(cluster_), "titles": title_names, "topic": json.loads(temp['topic'])}
