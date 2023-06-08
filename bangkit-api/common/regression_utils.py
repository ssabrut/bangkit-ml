import numpy as np
import tensorflow as tf


class RegressionModel:
    cities = list(map(lambda x: x.lower(), ['Pacitan', 'Ponorogo', 'Trenggalek', 'Tulungagung', 'Blitar',
                                            'Kediri', 'Malang', 'Lumajang', 'Jember', 'Banyuwangi',
                                            'Bondowoso', 'Situbondo', 'Probolinggo', 'Pasuruan', 'Sidoarjo',
                                            'Mojokerto', 'Jombang', 'Nganjuk', 'Madiun', 'Magetan', 'Ngawi',
                                            'Bojonegoro', 'Tuban', 'Lamongan', 'Gresik', 'Bangkalan',
                                            'Sampang', 'Pamekasan', 'Sumenep', 'Surabaya', 'Batu']))
    plants = list(map(lambda x: x.lower(), ['Cabai Rawit', 'Bawang merah', 'Bawang putih', 'Kentang', 'Kubis']))

    def __init__(self, data):
        self.data = data
        self.model = tf.keras.models.load_model('common/models/price_model.h5')

    def one_hot_data(self):
        for city in self.cities:
            if self.data['daerah'].values[0].lower() == city.lower():
                self.data[city] = 1
            else:
                self.data[city] = 0

        for plant in self.plants:
            if self.data['tanaman'].values[0].lower() == plant.lower():
                self.data[plant] = 1
            else:
                self.data[plant] = 0

    def pipeline(self):
        self.one_hot_data()
        self.data[['luas_panen', 'produksi']] = self.data[['luas_panen', 'produksi']].apply(np.log)
        self.data = self.data.drop(['daerah'], axis=1)
        self.data = self.data.drop(['tanaman'], axis=1)
        yhat = self.model.predict(self.data.values)
        return str(np.exp(yhat[0][0]))
