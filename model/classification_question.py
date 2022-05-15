import pickle
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ClassificationQuestion(object):
    def __init__(self, model_path="model/weights/classification_question.pkl"):
        self.model = pickle.load(open(model_path, 'rb'))
        self.vocab = ['trên', 'đường', 'vật', 'vật_thể', 'dưới', 'lòng_đường', 'ngoài', 'đường_không', 'lề_đường', 'trong', 'vỉa_hè', 'lề', 'đi', 'bộ', 'người', 'hè', 'phố', 'cho', 'biết', 'phía', 'bên', 'trái_vậy', 'trái', 'liệt_kê', 'kể', 'tên', 'tay_trái', 'một_vài', 'phải', 'bao_gồm', 'vật_thể_nào', 'tay_phải', 'tay', 'hướng', 'đằng', 'trước', 'trước_vậy', 'mặt', 'kể_vật', 'đối_diện', 'tất_cả', 'xung_quanh', 'quanh_đây', 'quanh', 'đây', 'gần', 'gần_đây', 'đây_vậy', 'đến', 'tới', 'gần_vậy', 'khoảng_cách', 'ngắn', 'xa', 'xa_vậy']
        self.stop_word = ["có", "gì", "ở", "nằm", "cái", "không", "những", "thứ", "vậy", "nào", "các", "gồm", "mọi", "đang"]
    
    def tfidf(self, data):
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30, stop_words = self.stop_word, ngram_range=(1, 2), vocabulary = self.vocab)
        tfidf_vect_ngram.fit(data)
        return tfidf_vect_ngram.transform(data)

    def preprocess(self, x):
        e=ViTokenizer.tokenize(str(x).lower())
        return self.tfidf([e])

    def predict(self, question):
        x = self.preprocess(question)
        return self.model.predict(x)