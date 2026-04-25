from flask import Flask, request, flash, session, jsonify
from flask import render_template, redirect, url_for,Response
from tensorflow.keras.models import load_model
import os
import sqlite3
import numpy as np
# from PIL import Image
import pandas as pd
from datetime import datetime
# import time
# import logging
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers
import numpy as np
import os
import cv2

app = Flask(__name__)
app.secret_key = 'supersecretkey'



def get_db_connection():
    try:
        conn = sqlite3.connect('mydb.db')
        cursor = conn.cursor()

        # Create the 'users' table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                datetime TEXT NOT NULL
            )
        ''')

        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        return None
    

    
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')

    return render_template('index.html')

@app.route('/prediction')
def prediction():
    if 'user_id' in session:
        return render_template('prediction.html')
    return redirect(url_for('loginpage'))

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        from datetime import datetime
        username = request.form['name']
        print(username)
        email = request.form['email']
        password = request.form['password']
        datetime_now = datetime.now().isoformat()

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password, datetime) VALUES (?, ?, ?, ?)',
                           (username, email, password, datetime_now))
            conn.commit()
            conn.close()
            return render_template('login.html',username=username,password=password)
        else:
            return jsonify({'status': 'failed', 'error': 'Database connection failed'})
    except Exception as e:
        print(f"Error in add_user: {e}")
        return jsonify({'status': 'failed', 'error': str(e)})

def validate(username, password):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()

            if user and user['password'] == password:
                return True, username
        return False, username
    except sqlite3.Error as e:
        print(f"Database error during validation: {e}")
        return False, username
    
@app.route('/login', methods=['POST', 'GET'])
def login():
    try:
        username = request.form['username']
        password = request.form['password']

        completion, username = validate(username, password)
        if completion:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
                user = cursor.fetchone()
                conn.close()

                if user:
                    session['user_id'] = user['id']
                    session['user_name']=username
                    session['logged_in'] = True
                    return redirect(url_for('index'))
            return  render_template('login.html', error = 'User not found')
        return render_template('login.html', error = 'Invalid credentials')
    except Exception as E:
        return render_template('login.html',error=E)

@app.route('/upload', methods=['POST'])
def upload_image():
    
    try:
        if 'imageFile' in request.files:
            file = request.files['imageFile']
            if file.filename != '':
                # Save the uploaded file to a desired location
                filename = os.path.join('static', 'image',file.filename)
                file.save(filename)
                import shutil

                # Source file path
                source_file = filename
                # Destination directory for copying the file
                destination_directory = './'

                # Destination path for moving the file
                #destination_path = 'image.jpg'

                # Copy the file to the destination directory
                shutil.copy(source_file, destination_directory)
                try:
                    os.remove('image.jpg')
                except:
                    pass    
    
                os.rename(file.filename,'image.jpg')
                result = predicts()

                return jsonify({'status': 'success', 'image_url': filename,'result': result})
        else:
            return jsonify({'status': 'error','error':"file Uploading Error" })
    except Exception as E :
        return jsonify({'status': 'error','error':E})




######################################################
###############################################



def load_and_preprocess_image(image_path, image_size):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
    
    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Assuming the model was trained on normalized images

    # Expand dimensions to match the input shape of the model
    image_array = tf.expand_dims(image_array, axis=0)
    
    # Normalize the image array
    return image_array




def predicts():
    try:
        names= ['Aloevera','Amla','Bhrami','Bringaraja','Coriender','Curry','Ekka','Hibiscus','Lemon','Mint','Neem','Papaya','Tulsi']
        
        loaded_model = load_model('model/model_inceptionv2_withaug500.h5')

        # detect_model = `("./model/model_inceptionv2_withaug500.h5") 
        img = "image.jpg"
        # result=prediction_img(img,detect_model,classNames)
        img_features=load_and_preprocess_image("image.jpg",224)
        img_features.shape

        pred_probs=loaded_model.predict(img_features)
        pred=np.argmax(pred_probs)
        

        result=names[pred]
        print(result)
        confidence_scor = pred_probs[0][pred]  # Assuming batch size of 1

        # Get the predicted class label
        result1 = names[pred]
        
        #########################################
        image = cv2.imread(img)
        image = cv2.resize(image, (224, 224))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([30, 40, 40])  # Lower bound for green color
        upper_bound = np.array([90, 255, 255])  # Upper bound for green color
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        segmented_leaves = segmented.astype(np.float32) / 255.0
        segmented_leaves = segmented_leaves[tf.newaxis, ...]
        model = load_model('model/model_inceptionv2_withaug500andsegmented.h5')
        predictions = model.predict(segmented_leaves)
        
        pred=np.argmax(predictions)
        confidence_score = predictions[0][pred]  # Assuming batch size of 1
        result2 = names[pred]
        print("Confidence Score:",confidence_score)
        ######################################
        if confidence_scor > 0.70 :
            result=result1
        elif confidence_scor < 0.50 and confidence_score > 0.60:
            result=result2
        elif confidence_scor > confidence_score:
            result=result1
        elif confidence_scor < confidence_score:
            result=result2
            
        # Print the result along with confidence score
        print(f"Predicted Class: {result2}")
        print(f"Confidence Score: {confidence_scor * 100:.2f}%")
        
        if confidence_score< 0.50 or confidence_scor< 0.40:
            print(confidence_score,confidence_scor)
            result= 'Confidence score is very low so you can provide a good quality leaf image only'
        return result
    except Exception as E :
        return jsonify({'status': 'error','result': E})

def predictssss():
    names= ['Aloevera','Amla','Bhrami','Bringaraja','Coriender','Curry','Ekka','Hibiscus','Lemon','Mint','Neem','Papaya','Tulsi']
    # model.save('model_inceptionv2_withaug500.h5')
    loaded_model = load_model('model/model_inceptionv2_withaug500andseg.h5')

    # detect_model = `("./model/model_inceptionv2_withaug500.h5") 
    img = "image.jpg"
    lower_bound = np.array([30, 40, 40])  # Lower bound for green color
    upper_bound = np.array([90, 255, 255])  # Upper bound for green color
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    upper_bound = np.array([90, 255, 255])  # Upper bound for green color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_leaves = cv2.bitwise_and(image, image, mask=mask)
    # plt.imshow(image)  # Show original image
    # plt.imshow(segmented_leaves)  # Show segmented leaves


    # input_image = load_img(input_image_path, target_size=(IMAGE_RES, IMAGE_RES))
    # input_image_array = img_to_array(input_image)
    input_image_array = segmented_leaves / 255.0  # Normalize the image
    input_image_array = input_image_array[tf.newaxis, ...]
    predictions = loaded_model.predict(input_image_array)
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class_name = names[predicted_class_index]
    return jsonify({'status': 'success','result': predicted_class_name})








@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = predicts()
        return jsonify({'status': 'success','result': result})
    except Exception as E :
        return jsonify({'status': 'error','result': E})


@app.route('/loginpage')
def loginpage():
    if 'user_id' in session:
        return render_template('prediction.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)

    return redirect(url_for('index'))

@app.route('/register')
def register():
    if 'user_id' in session:
        return render_template('home.html')
    elif 'admin' in session:
        return render_template('adminpage.html')
    return render_template('register.html')


if __name__ == "__main__":
    app.run(debug=True)
