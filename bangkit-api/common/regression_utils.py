import numpy as np
import tensorflow as tf


class RegressionModel:
    cities = list(map(lambda x: x.lower(), ['Pacitan', 'Ponorogo', 'Trenggalek', 'Tulungagung', 'Blitar',
                                            'Kediri', 'Malang', 'Lumajang', 'Jember', 'Banyuwangi',
                                            'Bondowoso', 'Situbondo', 'Probolinggo', 'Pasuruan', 'Sidoarjo',
                                            'Mojokerto', 'Jombang', 'Nganjuk', 'Madiun', 'Magetan', 'Ngawi',
                                            'Bojonegoro', 'Tuban', 'Lamongan', 'Gresik', 'Bangkalan',
                                            'Sampang', 'Pamekasan', 'Sumenep', 'Surabaya', 'Batu']))

    def __init__(self, data):
        self.data = data
        self.model = tf.keras.models.load_model('common/models/price_model.h5')

    def one_hot_data(self):
        for city in self.cities:
            if self.data['daerah'].values == city:
                self.data[city] = 1
            else:
                self.data[city] = 0

    def pipeline(self):
        self.one_hot_data()
        self.data[['luas_panen', 'produksi']] = self.data[['luas_panen', 'produksi']].apply(np.log)
        yhat = self.model.predict(self.data.values)
        return yhat
