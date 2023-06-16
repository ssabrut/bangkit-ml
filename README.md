<h1 align="center">Botani Plan</h1>
<p align="center">
  <img src="https://github.com/ssabrut/bangkit-ml/assets/53653797/98b03874-7efc-4868-a8d9-6254c539ae7a"/>
</p>

## Team Profile
1. (ML) M305DSY2355 – Devina Margarita – Universitas Pembangunan Nasional “Veteran” Jawa Timur
2. (ML) M159DSX0188 – Michael Eko Hartono – Universitas Ciputra
3. (CC) C058DSX2777 – M Fahmi Alfaris – Politeknik Negeri Banyuwangi 
4. (CC)  C151DKX3960 – Muhammad Helmi Yahya – Universitas Brawijaya 
5. (MD) A151DSX2280 – Raya Aldyen Dessario – Universitas Brawijaya 
6. (MD) A151DSX2429 – Elang Sotya Putra Dumipta – Universitas Brawijaya

## Botani Plan Machine learning
Theme : Agriculture

### Project Background
The agricultural industry in Indonesia makes a significant contribution to the national economy. However, novice farmers often experience difficulties in managing their farming business due to lack of knowledge and experience. Most workers in the agricultural sector have limited education, so solutions are needed to help them.

In this project, an application was designed that uses machine learning technology to provide recommendations for the most suitable plants for novice farmers based on weather conditions. This app also provides agricultural market price predictions to help farmers make better decisions in their farming business. In this project, we focused on novice farmers in the horticulture sector and selected several crops with high economic value, namely shallots, garlic, potatoes, cabbage and cayenne pepper.
With this application, it is hoped that novice farmers can manage their farming business more effectively, increase crop yields, and reduce the risk of failure.

### Machine Learning
In this project, we developed two machine learning models that play an important role in the application. The following is an explanation from the machine learning side of each model:
1. **Crop Recommendation Model**
    
    This model uses machine learning and data processing techniques to provide plant recommendations based on climatic factors. Climate data is processed and classified into several categories such as air temperature, air humidity, rainfall, and altitude of the planting location.

    In this model, we use an LSTM (Long Short-Term Memory) neural network that has been trained with pre-processed data. This model calculates the similarity between the input plant representation and all plants in the dataset using the cosine comparison technique. Based on these similarities, the model recommends plants that are most similar to inputs as recommendations for farmers.

2. **Price Prediction Model**

    This model uses a machine learning algorithm that uses historical price data, harvested area, production, and area to predict crop prices. The data is processed and divided into numerical and categorical features. Then, the Sequential model with several Dense layers is used to predict prices.

    This model was trained using SGD optimizer and MSE loss function for 200 epochs. After going through the training process, the model can make price predictions using test data. The prediction results are then evaluated using the mean squared error (MSE) So, this model can be used to predict crop prices based on the historical data entered.

### Our Dataset Link
1. **Recommendation**

    * https://jatim.bps.go.id/indicator/151/88/1/rata-rata-suhu-udara.html
    * https://jatim.bps.go.id/indicator/151/89/1/kelembaban-udara.html
    * https://karangploso.jatim.bmkg.go.id/index.php/profil/meteorologi/list-of-all-tags/analisis-bulanan-distribusi-curah-hujan-tahun-2022
    * https://jatim.bps.go.id/statictable/2018/10/23/1310/rata-rata-tinggi-wilayah-di-atas-permukaan-air-laut-dpl-menurut-pos-hujan-di-kabupaten-kota-di-provinsi-jawa-timur-2017.html

2. **Price Prediction**

    * https://siskaperbapo.jatimprov.go.id/
    * https://jatim.bps.go.id/statictable/2023/03/03/2453/hortikultura-horticulture---luas-panen-tanaman-buncis-string-bean-cabai-besar-big-chili-cabai-rawit-chili-cayenn-papper-menurut-kabupaten-kota-dan-jenis-tanaman-di-provinsi-jawa-timur-ha-2020-dan-2021.html
    * https://jatim.bps.go.id/statictable/2023/03/16/2529/luas-panen-tanaman-sayuran-cabai-besar-cabai-rawit-cabai-keriting-menurut-kabupaten-kota-dan-jenis-tanaman-di-provinsi-jawa-timur-ha-2021-dan-2022.html
    * https://jatim.bps.go.id/statictable/2023/03/16/2534/-produksi-tanaman-sayuran-bawang-daun-dan-bawang-merah-menurut-kabupaten-kota-dan-jenis-tanaman-di-provinsi-jawa-timurkuintal-2021-dan-2022.html
    * https://jatim.bps.go.id/statictable/2023/03/03/2468/hortikultura-horticulture---produksi-tanaman-bawang-daun-sacallion-dan-bawang-merah-shallot-menurut-kabupaten-kota-dan-jenis-tanaman-di-provinsi-jawa-timur-kuintal-2020-dan-2021.html

### Dataset Preview
1. **Recommendation**
    
    ![image](https://github.com/ssabrut/bangkit-ml/assets/53653797/ea457a34-1851-4c00-b11b-dd477dabeae3)

2. **Price Prediction**
    
    ![image](https://github.com/ssabrut/bangkit-ml/assets/53653797/07434136-0398-436d-a24e-a1c7773f4cb6)

### Model Performance
1. **Recommendation**
   
   ![image](https://github.com/ssabrut/bangkit-ml/assets/53653797/cbfe2db8-ade9-49b4-bb84-e67c278abd39)

2. **Price Prediction**

    ![image](https://github.com/ssabrut/bangkit-ml/assets/53653797/fc55f78c-cbad-4fbf-b553-1344574cc081)

### Notebook Model
1. **Recommendation**: https://github.com/ssabrut/bangkit-ml/blob/main/Recommendation%20System.ipynb
2. **Price Prediction**: https://github.com/ssabrut/bangkit-ml/blob/main/Price%20Prediction.ipynb

### How to Run Flask App
First thing first, you need to clone the repository
~~~
git clone https://github.com/ssabrut/bangkit-ml.git
~~~

After cloning the repository, we recommend using Python 3.8 or newer. After that install the install the required dependencies by following.
~~~
cd bangkit-ml
cd bangkit-api
pip install -r requirements.txt
~~~

After installing all required dependencies, you are ready to go! By running this script.
~~~
python app.py
~~~

### Technology and Tools Used
1. Python: The programming language used in the implementation of recommendation and price prediction models.
2. Pandas: A Python library used to manipulate and analyze data.
3. NumPy: A Python library used for math operations and array training.
4. Matplotlib and Seaborn: Python libraries used for data visualization, such as creating plots, histograms, and counterplots.
5. Scikit-learn: A Python library that provides a variety of machine learning algorithms, including data preprocessing, performance measurement, and data processing.
6. TensorFlow and Keras: Machine learning frameworks used to build and train neural network models.
7. LSTM (Long Short-Term Memory) Neural Networks: A type of layer in a neural network model used to model sequential data, such as time sequences.
8. MinMaxScaler: A preprocessing method used to normalize data on numeric features.
9. SGD (Stochastic Gradient Descent): Optimizer used in the training model process.
10. Mean Squared Error (MSE): The loss function used to measure the difference between the predicted value and the actual value in the model.
11. Flask: Framework Python yang ringan dan mudah digunakan untuk membangun API web. Flask dapat digunakan untuk membuat endpoint API yang dapat menerima input dan memberikan output berdasarkan model yang telah dilatih.
