import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_avg_temp_classification(value):
    if value >= 24.299 and value < 27.8:
        return 'low temp'
    elif value >= 27.8 and value < 28.6:
        return 'medium temp'
    elif value >= 28.6:
        return 'high temp'
    return 'very low temp'

def create_humidity_classification(value):
    if value < 45.:
        return 'too dry'
    elif value >= 45. and value <= 65.:
        return 'ideal'
    return 'too moist'

def create_rainfall_classification(value):
    if value > 0 and value < 100:
        return 'low rainfall'
    elif value >= 100 and value < 300:
        return 'medium rainfall'
    elif value >= 300 and value < 500:
        return 'high rainfall'
    return 'very high rainfall'

def create_altitute_classification(value):
    if value > 0 and value < 200:
        return 'low land'
    elif value >= 200 and value < 500:
        return 'hill'
    return 'high land'

class RecommederModel:
    def __init__(self, data):
        self.data = data
        self.anchor_data = pd.read_csv('common/data/recommendation_df.csv')
        self.model = tf.keras.models.load_model('common/models/recommendation_model.h5')

    def preprocess(self):
        self.data['avg_temp'] = self.data['avg_temp'].apply(create_avg_temp_classification)
        self.data['humid'] = self.data['humid'].apply(create_humidity_classification)

    def pipeline(self):
        self.preprocess()
        test_data = ' '.join(self.data.values[0])
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.anchor_data['feature'])
        tfidf = tfidf_matrix.toarray()
        plant_representation = self.model.predict(tfidf)
        tfidf_matrix = vectorizer.transform([test_data])
        tfidf = tfidf_matrix.toarray()
        test_representation = self.model.predict(tfidf)
        similar_plants = cosine_similarity(test_representation, plant_representation)
        recommended_plant_index = tf.argmax(similar_plants)
        recommended_plant = self.anchor_data.iloc[recommended_plant_index]['plant_clean'].drop_duplicates()
        return recommended_plant.values[0].split()



