import os
import shutil
from datetime import date, datetime
import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.neighbors import KNeighborsClassifier

# Defining Flask App
app = Flask(__name__)

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# Get the number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# Extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


def identify_face(facearray):
    if not os.path.isfile('static/face_recognition_model.pkl'):
        return "Unrecognized"

    model = joblib.load('static/face_recognition_model.pkl')
    prediction = model.predict(facearray)
    return prediction[0] if prediction else "Unrecognized"




# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names, rolls, times = df['Name'], df['Roll'], df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')[0], name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in df['Roll'].values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)



# This function will run when we click on the Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    names, rolls, times, l = extract_attendance()  # Define and initialize 'names' variable
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            resized_face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(resized_face.reshape(1, -1))
            if identified_person == "Unrecognized":
                cv2.putText(frame, "Unrecognized", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
            else:
                add_attendance(identified_person)
                cv2.putText(frame, identified_person, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)



# This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while i < 50:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])
                i += 1
            j += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/delete', methods=['POST'])
def delete():
    try:
        username = request.form['username']
        userid = request.form['userid']

        # delete user's images
        user_folder = os.path.join('static', 'faces', f'{username}_{userid}')
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)

        # delete user's attendance records
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        df = df[df['Roll'] != int(userid)]
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

        # delete user's data from the model
        userlist = os.listdir('static/faces')
        user_image_paths = {}
        for user in userlist:
            if user != f'{username}_{userid}':
                user_image_paths[user] = [
                    os.path.join('static', 'faces', user, imgname)
                    for imgname in os.listdir(f'static/faces/{user}')
                ]
        knn = KNeighborsClassifier(n_neighbors=5)
        if user_image_paths:
            faces, labels = [], []
            for user, image_paths in user_image_paths.items():
                for img_path in image_paths:
                    img = cv2.imread(img_path)
                    resized_face = cv2.resize(img, (50, 50))
                    faces.append(resized_face.flatten())
                    labels.append(user)
            faces = np.array(faces)
            knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        if not os.listdir('static/faces'):
            os.remove('static/face_recognition_model.pkl')

        return redirect(url_for('home'))
    except Exception as e:
        return f'Error: {str(e)}'


if __name__ == '__main__':
    app.run(debug=True)
