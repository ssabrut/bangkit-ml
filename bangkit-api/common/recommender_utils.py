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

def preprocess_plant(values):
    combined = ''
    for value in values.split(','):
        if len(value.split()) > 1:
            value = '-'.join(value.split())
        if value == 'caba':
            value = 'cabai'
        if value == 'ccabai':
            value = 'cabai'
        if value == 'mruah':
            value = 'bawang-merah'
        if value == 'cabang':
            value = 'cabai'
        if value == 'kenrang':
            value = 'kentang'
        combined += value + ' '
    return combined.strip()

def combine_area(values):
    combined = ''
    for value in values.split(','):
        combined += value + ' '
    return combined.strip()

class RecommederModel:
    def __init__(self, data):
        self.data = data
        self.anchor_data = pd.read_csv('common/data/recommendation_df.csv')
        self.model = tf.keras.models.load_model('common/models/recommendation_model.h5')

    def preprocess(self):
        self.data['avg_temp_classification'] = create_avg_temp_classification(self.data['avg_temp'])
        self.data['humid_classification'] = create_humidity_classification(self.data['percent_humid'])
        self.data['avg_rainfall_classification'] = create_rainfall_classification(self.data['avg_rainfall'])
        self.data['altitude_classification'] = create_altitute_classification(self.data['altitude'])
        final_df = pd.concat([
            self.data.loc[:, ['avg_temp_classification']],
            self.data.loc[:, ['humid_classification']],
            self.data.loc[:, ['avg_rainfall_classification']],
            self.data.loc[:, ['altitude_classification']],
            self.data.loc[:, ['area']]
        ], axis=1)
        return final_df.iloc[0].agg(' '.join, axis=1)

    def pipeline(self):
        test_data = self.preprocess()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.anchor_data['feature'])
        tfidf = tfidf_matrix.toarray()
        plant_representation = self.model.predict(tfidf)
        tfidf_matrix = vectorizer.transform([test_data])
        tfidf = tfidf_matrix.toarray()
        test_representation = self.model.predict(tfidf)
        similar_plants = cosine_similarity(test_representation, plant_representation)
        return similar_plants



