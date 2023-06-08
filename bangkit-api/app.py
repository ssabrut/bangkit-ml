import pandas as pd
from flask import Flask
from flask_restful import Resource, Api
from flask import jsonify
from flask import request
from common.recommender_utils import RecommederModel
from common.regression_utils import RegressionModel

app = Flask(__name__)
api = Api(app)


class RecommendationApi(Resource):
    def post(self):
        try:
            _json = request.json
            _temp = _json['avg_temp']  # float
            _humid = _json['humid']  # float
            _rainfall = _json['rainfall']  # low rainfall / medium rainfall / high rainfall / very high rainfall
            _altitude = _json['altitude']  # low land / hill / high land
            _city = _json['city']

            with app.app_context():
                data = pd.DataFrame(
                    [[_temp, _humid, _rainfall, _altitude, _city]],
                    columns=['avg_temp', 'humid', 'rainfall', 'altitude', 'city']
                )

                recommender_model = RecommederModel(data=data)
                if recommender_model:
                    return jsonify({
                        'status_code': 200,
                        'message': 'success getting plant recommendation',
                        'recommendation': recommender_model.pipeline()
                    })
                else:
                    return jsonify({
                        'status_code': 400,
                        'message': 'failed to initialize model'
                    })
        except Exception as e:
            return jsonify({
                'status_code': 500,
                'message': str(e)
            })


class RegressionApi(Resource):
    def post(self):
        try:
            _json = request.json
            _luas_panen = _json['luas_panen']  # float
            _produksi = _json['produksi']  # float
            _daerah = _json['daerah']
            _tanaman = _json['tanaman'] # ['Cabai Rawit', 'Bawang merah', 'Bawang putih', 'Kentang', 'Kubis']

            with app.app_context():
                data = pd.DataFrame(
                    [[_luas_panen, _produksi, _daerah, _tanaman]],
                    columns=['luas_panen', 'produksi', 'daerah', 'tanaman']
                )

                regression_model = RegressionModel(data=data)
                if regression_model:
                    return jsonify({
                        'status_code': 200,
                        'message': 'success getting plant price prediction',
                        'recommendation': regression_model.pipeline()
                    })
                else:
                    return jsonify({
                        'status_code': 400,
                        'message': 'failed to initialize model'
                    })
        except Exception as e:
            return jsonify({
                'status_code': 500,
                'message': str(e)
            })


api.add_resource(RecommendationApi, '/get-recommendation')
api.add_resource(RegressionApi, '/get-price-prediction')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
