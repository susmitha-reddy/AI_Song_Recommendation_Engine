import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import requests
import json
from csv import writer
import pandas as pd
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\PAVANI\\Documents\\Project\\ui\\uploaded\\image\\'

#loading the already saved CNN model for face recognition and Random Forest model for Song classification

model_randomforest = pickle.load(open('randomforest_songclassifier.pickle', 'rb'))
model_cnn = keras.models.load_model('CNN_face_classifier.h5')

#trying to integrate webapplication with the spotify API

client_id = '745bc2161f844b48b864f695fe7432fb'
client_secret = '7a2f12f6bbf043fc9dca9e3b327402bc'
grant_type = 'client_credentials'

#Request based on Client Credentials Flow from https://developer.spotify.com/web-api/authorization-guide/
#Request body parameter: grant_type Value: Required. Set it to client_credentials

body_params = {'grant_type' : grant_type}
url='https://accounts.spotify.com/api/token'
response=requests.post(url, data=body_params, auth = (client_id, client_secret)) 
res  = response.text
res1 = eval(res)
token = res1["access_token"]
headers = {'Authorization': 'Bearer '+ token}

#Function to get the spotify audio features of the track by taking the spotify track id as input
def get_song_features(id):
    url1="https://api.spotify.com/v1/audio-features/"+ str(id)
    url2 = "https://api.spotify.com/v1/tracks/"+ str(id)
    r1 = requests.get(url1, headers=headers)
    r2 = requests.get(url2, headers=headers)
    name_info = r2.json()
    feature_info = r1.json()
    albumname = name_info['album']['name']
    artistname = name_info['album']['artists'][0]['name']
    songname= name_info['name']
    features = list([feature_info['danceability'] , feature_info['acousticness'], feature_info['energy'], feature_info['instrumentalness'], feature_info['valence'],feature_info['liveness'], feature_info['loudness'],feature_info['speechiness'], feature_info['tempo']])
    names = list([songname,albumname,artistname])
    return features, names

@app.route('/')
def home():
    return render_template('index.html')

# function to display playlist based on the emotion of the input image
@app.route('/recommend',methods=['POST'])
def recommend():
    if request.method == 'POST':
        f = request.files['file']
        print("pavani", f)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        path = app.config['UPLOAD_FOLDER'] + secure_filename(f.filename)
        print("pavani", path)
        img = cv2.imread(path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("pavani", gray_img.shape)
        img = np.expand_dims(gray_img,axis = 0) #makes image shape (1,48,48)
        print("pavani", img.shape)
        img = img.reshape(1,48,48,1)
        result = model_cnn.predict(img)   #saved CNN model is used to predict the emotion of the input image
        result = list(result[0])
        pos = result.index(max(result)) 
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        expression = label_dict[pos]
        df_merged_final_list = pd.read_csv('C:\\Users\\PAVANI\\Documents\\Project\\file_songs.csv')
        df_happy = df_merged_final_list[df_merged_final_list['mood'] == 'Happy']
        df_sad = df_merged_final_list[df_merged_final_list['mood'] == 'Sad']
        df_calm= df_merged_final_list[df_merged_final_list['mood'] == 'Calm']
        df_energetic = df_merged_final_list[df_merged_final_list['mood'] == 'Energetic']
        
        if(expression == 'Happy'):
            df_result= pd.DataFrame(df_happy.sample(n = 10))
        elif(expression == 'Sad' or expression == 'Disgust'):
            df_result= pd.DataFrame(df_sad.sample(n = 10))
        elif(expression == 'Neutral' or expression == 'Surprise'):
            df_result= pd.DataFrame(df_energetic.sample(n=10))
        else:
            df_result = pd.DataFrame(df_calm.sample(n=10))
        list_Songs = df_result.values.tolist()
        return render_template('list.html',filename = path,expression=expression,list_Songs=list_Songs)

#function to add new track to the existing playlist 
@app.route('/song_add',methods=['POST'])
def song_add():
    prediction_text='sucessfully added'
    df = pd.read_csv('C:\\Users\\PAVANI\\Documents\\Project\\file_songs.csv')
    try:
        song_id = request.form.get("songid")
        list_id = df['id'].tolist()
        if(song_id in list_id):
            prediction_text = 'Song Already Exists'
            return render_template('index.html', prediction_text = prediction_text)
        
        audiofeatures, names = get_song_features(song_id)
        X_test = np.array(audiofeatures).reshape(1,-1)
        prediction = model_randomforest.predict(X_test)
        if(prediction == 0):
           mood = 'Calm'
        elif(prediction == 1):
           mood = 'Energetic'
        elif(prediction == 2):
           mood = 'Happy'
        else:
           mood = 'Sad'
        
        list_to_add = [song_id, names[0], mood]
        print(list_to_add)
        with open('C:\\Users\\PAVANI\\Documents\\Project\\file_songs.csv', 'a',newline='') as f_object:
  
            writer_object = writer(f_object)

            writer_object.writerow(list_to_add)

            f_object.close()
    except:
        prediction_text='Some error occured while adding'
           
    return render_template('index.html', prediction_text = prediction_text)

if __name__ == "__main__":
    app.run(debug=True)