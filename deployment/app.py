import numpy as np
from flask import Flask, request, render_template
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('templates/model_RF.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    data = pd.read_excel('New_Data.xlsx',engine='openpyxl')
    X_samp = data.copy()
    Y_samp = data['Churn Value']
    To_drop = ['Churn Value', 'CLTV', 'Churn Score']
    X_samp = X_samp.drop(To_drop, axis=1)
    X=X_samp
    Y=Y_samp
    Features_model_RF = X.copy()
    All_features = ['Latitude', 'Longitude', 'Gender_Male', 'Gender_Female', 'Multiple Lines_No phone service',
                    'Phone Service', 'Multiple Lines_No', 'Multiple Lines_Yes', 'Streaming Movies_Yes',
                    'Streaming TV_Yes', 'Device Protection_Yes', 'Online Backup_Yes', 'Payment Method_Mailed check',
                    'Payment Method_Bank transfer (automatic)', 'Internet Service_DSL', 'Streaming TV_No',
                    'Streaming Movies_No', 'Payment Method_Credit card (automatic)', 'Partner', 'Senior Citizen_No',
                    'Senior Citizen_Yes', 'Tech Support_Yes', 'Online Security_Yes', 'Contract_One year',
                    'Paperless Billing', 'Monthly Charges', 'Total Charges', 'Device Protection_No internet service',
                    'Streaming Movies_No internet service', 'Streaming TV_No internet service',
                    'Tech Support_No internet service', 'Online Security_No internet service',
                    'Online Backup_No internet service', 'Internet Service_No', 'Dependents', 'Device Protection_No',
                    'Online Backup_No', 'Payment Method_Electronic check', 'Contract_Two year',
                    'Internet Service_Fiber optic', 'Tech Support_No', 'Online Security_No', 'Tenure Months',
                    'Contract_Month-to-month']
    To_drop = All_features[:(len(Features_model_RF.columns) - 43)]
    Features_model_RF = Features_model_RF.drop(To_drop, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(Features_model_RF, Y, train_size=0.75, random_state=3,
                                                        stratify=Y)
    robust = RobustScaler()
    X_train = robust.fit_transform(X_train)

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    final_features = robust.transform(final_features)
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)