from flask import Flask, render_template, request, flash, redirect

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,flash,redirect, url_for, session,flash
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'
# Load the trained machine learning model
model = load_model('my_model.h5')


# Load the scaler object used during training
scaler = StandardScaler()
scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)


def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        durl="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d30980.77556739814!2d75.54684087061477!3d13.923023340574629!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1sdiabetics%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712761346961!5m2!1sen!2sin"
        return (model.predict(values.reshape(1, -1))[0],durl)
    
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        durl="https://www.google.com/maps/embed?pb=!1m12!1m8!1m3!1d30980.775335539816!2d75.54684088582736!3d13.92302507028802!3m2!1i1024!2i768!4f13.1!2m1!1sheart%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712761261910!5m2!1sen!2sin"
        return (model.predict(values.reshape(1, -1))[0],durl)
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        durl="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d30980.77579925653!2d75.54684085540214!3d13.92302161086144!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1skidney%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712761425519!5m2!1sen!2sin"
        return (model.predict(values.reshape(1, -1))[0],durl)
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        durl="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d30980.776031114867!2d75.54684084018953!3d13.923019881148454!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1sliver%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712761479618!5m2!1sen!2sin"
        return (model.predict(values.reshape(1, -1))[0],durl)

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred,durl = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred,doct=durl)


@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred,doct="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d247785.78143718545!2d75.29231728158405!3d13.979258182611968!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1spnemonia%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712760834392!5m2!1sen!2sin")


# Define the form page route
@app.route('/lungs')
def lungs():
    return render_template('lungcancer.html')
    
@app.route('/diabetes_about')
def diabetes_about():
    return render_template('diabetes_about.html')
    
@app.route('/heart_about')
def heart_about():
    return render_template('heart_about.html')

@app.route('/kidney_about')
def kidney_about():
    return render_template('kidney_about.html')


@app.route('/liver_about')
def liver_about():
    return render_template('liver_about.html')

@app.route('/lung_cancer_about')
def lung_cancer_about():
    return render_template('lung_cancer_about.html')

@app.route('/pneumonia_about')
def pneumonia_about():
    return render_template('pneumonia_about.html')


@app.route('/lung', methods=['GET', 'POST'])
def lung():
    if request.method == 'POST':
        # Print out the form data for debugging
        print(request.form)

        # Extract the form data
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        if gender == 'Male': gender = 1
        else: gender = 2
        air_pollution = int(request.form.get('air_pollution'))
        alcohol_use = int(request.form.get('alcohol_use'))
        dust_allergy = int(request.form.get('dust_allergy'))
        occupational_hazards = int(request.form.get('occupational_hazards'))
        genetic_risk = int(request.form.get('genetic_risk'))
        chronic_lung_disease = int(request.form.get('chronic_lung_disease'))
        balanced_diet = int(request.form.get('balanced_diet'))
        obesity = int(request.form.get('obesity'))
        smoking = int(request.form.get('smoking'))
        passive_smoker = int(request.form.get('passive_smoker'))
        chest_pain = int(request.form.get('chest_pain'))
        coughing_blood = int(request.form.get('coughing_blood'))
        fatigue = int(request.form.get('fatigue'))
        weight_loss = int(request.form.get('weight_loss'))
        shortness_of_breath = int(request.form.get('shortness_of_breath'))
        wheezing = int(request.form.get('wheezing'))
        swallowing_difficulty = int(request.form.get('swallowing_difficulty'))
        clubbing = int(request.form.get('clubbing'))
        frequent_cold = int(request.form.get('frequent_cold'))
        dry_cough = int(request.form.get('dry_cough'))
        snoring = int(request.form.get('snoring'))

        data = np.zeros((23))
        data[0] = age
        data[1] = gender
        data[2] = air_pollution
        data[3] = alcohol_use
        data[4] = dust_allergy
        data[5] = occupational_hazards
        data[6] = genetic_risk
        data[7] = chronic_lung_disease
        data[8] = balanced_diet
        data[9] = obesity
        data[10] = smoking
        data[11] = passive_smoker
        data[12] = chest_pain
        data[13] = coughing_blood
        data[14] = fatigue
        data[15] = weight_loss
        data[16] = shortness_of_breath
        data[17] = wheezing
        data[18] = swallowing_difficulty
        data[19] = clubbing
        data[20] = frequent_cold
        data[21] = dry_cough
        data[22] = snoring

        # Convert the list to a numpy array with the desired shape
        new_data = np.array([data])

        # Standardize the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)

        # Make predictions
        predictions = model.predict(new_data_scaled)

        # Convert the predictions to class labels
        predicted_classes = np.argmax(predictions, axis=1)


        # Determine the predicted outcome based on the prediction
        if predicted_classes[0] == 0:
            outcome = 1
        elif predicted_classes[0] == 1:
            outcome = 0
        else:
            outcome = 0

        # Render the results template with the predicted outcome
        return render_template('predict.html', pred=outcome,doct="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d30980.77626297315!2d75.54684082497691!3d13.923018151435647!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1slungs%20treatment%20shivamogga!5e0!3m2!1sen!2sin!4v1712761556525!5m2!1sen!2sin")
    else:
        # If the request method is not POST, redirect to the home page
        return redirect('/')

if __name__ == '__main__':
	app.run(debug = True)
