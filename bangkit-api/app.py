import pandas as pd
from flask import Flask
from flask_restful import Resource, Api
from flask import jsonify
from flask import request
from common.recommender_utils import RecommederModel

app = Flask(__name__)
api = Api(app)


class RecommendationApi(Resource):
    def get(self):
        try:
            _json = request.json
            _temp = _json['avg_temp']
            _humid = _json['humid']
            _rainfall = _json['rainfall']
            _altitude = _json['altitude']
            _city = _json['city']

            with app.app_context():
                data = pd.DataFrame({
                    'avg_temp': _temp,
                    'humid': _humid,
                    'rainfall': _rainfall,
                    'altitude': _altitude,
                    'city': _city
                })

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


api.add_resource(RecommendationApi, '/get-recommendation')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
