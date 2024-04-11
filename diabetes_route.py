from flask import Blueprint, request,jsonify
import numpy as np
import pickle


diabetes_api = Blueprint('diabetes_api',__name__)
model = pickle.load(open("diabetes_model.sav", 'rb'))


# Register route
@diabetes_api.route('/diabetes', methods=['POST'])
def diabetes_check():
    try:
        # Extract data from request
        data = request.json
        age = data.get('age')
        hypertension = data.get('hypertension')
        heart_disease = data.get('heart_disease')
        bmi = data.get('bmi')
        hba1c_level = data.get('hba1c_level')
        glucose_level = data.get('glucose_level')
        gender = data.get("gender")
        smoking_history = data.get("smoking_history")

        input_data = (age,hypertension,heart_disease,bmi,hba1c_level,glucose_level,gender,smoking_history)
        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)

        if (prediction[0] == 0):
          
            return jsonify({'message': 'The person is not diabetic'}), 201
        else:
            
            return jsonify({'message': 'The person is diabetic'}), 201


        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500