from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import pandas as pd


model = KMeans()
svd = TfidfVectorizer()
vectorizer = TruncatedSVD()
cluster_ = pd.DataFrame()

def load_model():
    with open('models/Model_005.bin', 'rb') as f:
        global model

        model = pickle.load(f)

    with open('models/Model_005_vectorizer.bin', 'rb') as f:
        global vectorizer

        vectorizer = pickle.load(f)

    with open('models/Model_005_svd.bin', 'rb') as f:
        global svd

        svd = pickle.load(f)


def load_data():
    global cluster_
    cluster_ = pd.read_csv('datas/Model_005_cluster.csv')

def get_cluster(keyword: str):
    prediction = model.predict(svd.transform(vectorizer.transform([keyword])))
    return {
        "cluster": prediction[0],
        "topic": cluster_['top_ten'].iloc[prediction[0]],
    }
