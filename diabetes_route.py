from flask import Blueprint, request,jsonify
import numpy as np
import pickle

heart_api = Blueprint('heart_api',__name__)

model = pickle.load( open("diabetes_model.sav", 'rb'))


# Register route
@heart_api.route('/heart_disease', methods=['POST'])
def diabetes_check():
    try:
        # Extract data from request
        data = request.json
        age = data.get('age')
        sex = data.get('sex')
        cp = data.get('cp')
        chol = data.get('chol')
        fbs = data.get('fbs')
        restecg = data.get('restecg')
        thalach = data.get("thalach")
        exang = data.get("exang")
        oldpeak = data.get('oldpeak')
        slope = data.get("slope")
        ca = data.get("ca")
        thal = data.get("thal")


        input_data = (age,sex,cp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)



        if (prediction[0]== 0):
            print('The Person does not have a Heart Disease')
        else:
            print('The Person has Heart Disease')
    except Exception as e:
        return jsonify({'error': str(e)}), 500