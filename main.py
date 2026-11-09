from flask import Flask, render_template, request, redirect, url_for,jsonify,session,flash,Response,send_file
from werkzeug.utils import secure_filename
import pandas as pd
import os
import requests
import re
import pandas as pd
from io import StringIO
import tensorflow as tf
import io;
import matplotlib.pyplot as plt
import os
import numpy as np;
from keras.preprocessing.image import img_to_array
from keras.utils import load_img
from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras import backend as K
from keras import models
from sklearn import metrics
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns
import base64



app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.static_folder = 'static'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
df_train = pd.DataFrame()  
df_test = pd.DataFrame()   
y_test=np.array([])
evulation=None
y_train=np.array([])
X_train= np.array([])
X_test=np.array([])
X_val= np.array([])
y_val=np.array([])
evaluation=None
conf_matrix=np.array([])
accuracy1=None
precision=None
recall=None
f1=None
train_shape=None
test_shape=None
file_path_train=None
file_path_test=None
accuracy=np.array([])
loss=np.array([])
val_accuracy=np.array([])
val_loss=np.array([])
epoch=np.array([])
model=None

#définit les extensions des fichiers possible
def allowed_url(url):
    valid_extensions = {'xls', 'xlsx', 'csv', 'zip'}
    filename = url.split('/')[-1]
    extension = filename.split('.')[-1].lower()
    return extension in valid_extensions

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xls', 'xlsx', 'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#home route
@app.route('/')
def index():
    return render_template('home.html', filename=None)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global file_path_train,file_path_test
    if request.method == 'POST':
        upload_type = request.form.get('upload_type')

        if upload_type == 'file':
            # Récupérer le type de téléchargement à partir des données du formulaire
            file_train = request.files['file']
            file_test = request.files['file1'] 
            
            if file_train and file_test:
                if file_train.filename == '' or file_test.filename == '':
                    #renvoyer un message d'erreur au modèle upload.html si les deux fichiers sont vides
                    return render_template('upload.html', message="Veuillez sélectionner les deux fichiers.")
        
        if file_train and file_test and allowed_file(file_train.filename) and allowed_file(file_test.filename):

            filename_train = secure_filename(file_train.filename)
            filename_test = secure_filename(file_test.filename)

            # Définir les chemins des fichiers pour enregistrer les fichiers téléchargés
            file_path_train = os.path.join(app.config['UPLOAD_FOLDER'], file_train.filename)
            file_path_test = os.path.join(app.config['UPLOAD_FOLDER'], file_test.filename)
            
            file_train.save(file_path_train)
            file_test.save(file_path_test)
            
            # Rendre le modèle upload.html avec un message de succès et les noms de fichiers
            return render_template('upload.html', message="Fichiers téléchargés avec succès.", filename_train=filename_train, filename_test=filename_test)
        
        # Si des fichiers sont manquants ou non autorisés, renvoyer un message d'erreur
        return render_template('upload.html', message="Veuillez télécharger les fichiers CSV uniquement.")

    return render_template('upload.html')


@app.route('/afficherShape')
def get_shape():
    global file_path_train,file_path_test
    if file_path_train is not None:
        df_train = pd.read_csv(file_path_train)
        df_test = pd.read_csv(file_path_test)
        train_shape = df_train.shape
        test_shape = df_test.shape
        return jsonify({
            'train_shape': train_shape,
            'test_shape': test_shape
        })
    return  render_template('upload.html')
    

@app.route('/display', methods=['POST'])
def display_data():
    global df_train, df_test, X_test,y_test,X_train,y_train,train_shape,test_shape

    # Récupérer les données
    filename_train = request.form.get('filename_train')
    filename_test = request.form.get('filename_test')
    num_rows = int(request.form.get('num_rows'))
    num_cols = int(request.form.get('num_cols'))

    if not filename_train or not filename_test:
        return jsonify({'error': 'Fichier(s) manquant(s)'}), 400

    file_path_train = os.path.join(app.config['UPLOAD_FOLDER'], filename_train)
    file_path_test = os.path.join(app.config['UPLOAD_FOLDER'], filename_test)

    if not os.path.exists(file_path_train) or not os.path.exists(file_path_test):
        return jsonify({'error': 'Fichier(s) introuvable(s)'}), 404

    # Lire les jeux de données d'entraînement et de test depuis les fichiers CSV 
    df_train = pd.read_csv(file_path_train)
    df_test = pd.read_csv(file_path_test)

    # Séparer les étiquettes et les fonctionnalités des jeux de données
    y_train = df_train.iloc[:,0:1].values
    X_train = df_train.iloc[:, 1:].values
    y_test = df_test.iloc[:,0:1].values
    X_test = df_test.iloc[:, 1:].values

    # Filtrer les jeux de données en fonction des entrées utilisateur pour le nombre de lignes et de colonnes à afficher
    df_train_display = df_train.head(num_rows)
    df_test_display = df_test.head(num_rows)


    
    #un condition pour s'assurer que le nombre de columns entrer par l'urilisateur est inférieur à le nombre total de data set
    if num_cols < len(df_train.columns):
        df_train_display = df_train_display.iloc[:, :num_cols]
        df_test_display = df_test_display.iloc[:, :num_cols]

    # Retourner les jeux de données filtrés au format JSON, ainsi que leurs dimensions
    return jsonify({
        'df_train': df_train_display.to_dict(orient='records'),
        'df_test': df_test_display.to_dict(orient='records'),
        
    })




def train_model(num_filt,num_filt2, kernel_size, pool_size,optimizer,loss,batch_size,epochs):
    global df_train, df_test, y_test,evaluation,y_train,X_test,X_train,X_train,model,X_val,y_val

    print("Shape of df_train before modification:", df_train.shape)
    print("Shape of df_test before modification:", df_test.shape)
    print("Shape of df_train before modification:", df_train.shape)
    print("Columns of df_train:", df_train.columns)
    print("Head of df_train:\n", df_train.head())

    #Prétraitement des données d'entrée (mise à l'échelle/Normalisation et remodelage/transformer les données d'image en un format approprié pour l'entrée du modèle CNN)
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)
    print("Example X_test:", X_test[:1])
    print("Example y_test:", y_test[:1])
    print(y_train)
    print(X_test)

    #encodage one-hot 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Division des données d'entraînement en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Définition du modèle CNN
    model = Sequential([
        Conv2D(num_filt, kernel_size= kernel_size, strides=(1,1), activation='relu' ,input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    
        Conv2D(num_filt2, kernel_size= kernel_size,activation='relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),


        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation="softmax")
    ])

    # Compilation du modèle
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Entraînement du modèle
    model_fitting = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, y_val))

    #  Évaluation du modèle sur l'ensemble de test pour mesurer les performances et la capacité à généraliser à de nouvelles données non vues lors de l'entraînement
    model.evaluate(X_test, y_test, verbose=1)

    return model_fitting
    

@app.route('/cnn', methods=['GET', 'POST'])
def train_model_route():

    if request.method == 'POST':
        global df_test, df_train, y_test, evaluation, y_train, conf_matrix, accuracy1, precision, recall, f1,X_train,X_test,Y_pred,accuracy,val_accuracy,loss,val_loss

        # Récupérer les paramètres du formulaire
        num_filt = int(request.form.get('num_filt'))
        kernel_size = int(request.form.get('kernel_size'))
        pool_size = int(request.form.get('pool_size'))
        epochs = int(request.form.get('epochs'))
        num_filt2 = int(request.form.get('num_filt2'))
        optimizer = request.form.get('optimizer')
        loss = request.form.get('loss')
        batch_size = int(request.form.get('batch_size'))

        # Entraîner le modèle avec les paramètres choisis par l'utilisateur
        trained_model = train_model(num_filt, num_filt2, kernel_size, pool_size, optimizer,
                                    loss, batch_size, epochs)

        if trained_model:
            model = trained_model.model
            print("Model input shape:", model.input_shape)
            
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
            print("Example X_test:", X_test[:1])
            print("Example y_test:", y_test[:1])

            # Prédire les classes sur l'ensemble de test
            Y_pred = model.predict(X_test)
            Y_pred = np.argmax(Y_pred, axis=1)
            y_test = np.argmax(y_test,axis = 1) 


            accuracy=trained_model.history["accuracy"]
            val_accuracy=trained_model.history["val_accuracy"]
            val_loss=trained_model.history["val_loss"]
            loss=trained_model.history["loss"]


            # Calculer les métriques de performance du modèle
            conf_matrix = metrics.confusion_matrix(y_test, Y_pred)
            accuracy1 = metrics.accuracy_score(y_test, Y_pred)
            precision = metrics.precision_score(y_test, Y_pred, average='macro')
            recall = metrics.recall_score(y_test, Y_pred, average='macro')
            f1 = metrics.f1_score(y_test, Y_pred, average='macro')

            return render_template('cnn.html', accuracy=accuracy1, precision=precision, recall=recall, f1=f1)

    return render_template('cnn.html')

def generer_heatmap_image(matrice_confusion):

    plt.figure(figsize=(11, 6))
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Purples')
    plt.xlabel('Étiquettes prédites')
    plt.ylabel('Étiquettes réelles')
    plt.title('Matrice de confusion')
    plt.tight_layout()

    # Enregistrer le plot comme une image PNG
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encodage de l'image en base64 pour l'intégrer dans une page web
    heatmap_image = base64.b64encode(img.getvalue()).decode()
    plt.close() # Fermeture de la figure pour libérer la mémoire

    return heatmap_image

@app.route('/matrice_confusion')
def confusion_matrix_data():
    global conf_matrix

    # Générer une image de la matrice de confusion
    confusion_Image = generer_heatmap_image(conf_matrix)

    # Convertir l'image encodée en base64 en fichier PNG et puis envoyer l'image
    return send_file(io.BytesIO(base64.b64decode(confusion_Image)),mimetype='image/png')


@app.route('/Accuracy_Class')
def get_data():
    global conf_matrix

    #calculer l'accuracy pour chaque classe de jeu de données
    class_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1)

    labels=["T-shirt/top", "Pantalon", "Pull", "Robe", "Manteau", "Sandale", "Chemise", "Basket", "Sac", "Bottine"]
    data = {
        'class_accuracy': class_accuracy.tolist(),
        'class_names': labels
    }
    return jsonify(data)


@app.route('/data')
def get_evulation():
    global accuracy,val_accuracy

    history = {
    "accuracy": accuracy,
    "val_accuracy":val_accuracy
    }
    print(history)
    return jsonify(history)

@app.route('/lossVal')
def get_loss():
    global val_loss,loss
    history2 = {
    "loss": loss,
    "val_loss":val_loss
    }
    print(history2)
    return jsonify(history2)

@app.route('/cnn', methods=['GET'])
def show_heatmap():
    return render_template('cnn.html')


def allowed_image(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif',}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_image(filename):
    # Ouvrir l'image 
    img = Image.open(filename)

    # Redimensionner l'image à une taille de 28x28 pixels
    img = img.resize((28, 28))
    # Convertir l'image en niveau de gris
    img = img.convert('L')
    
    # Convertir l'image en un tableau numpy et normaliser les valeurs des pixels entre 0 et 1
    img_array = np.array(img) 
    img_array = img_array.reshape((1, 28, 28, 1))/ 255.0

    return img_array

@app.route("/predict", methods=['GET','POST'])
def predict():
    global model

    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_image(file.filename):
            # Sauvegarder le fichier dans un emplacement temporaire
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Lire et prétraiter l'image
            img = read_image(file_path)

            # Prédire les probabilités de classe
            class_probabilities = model.predict(img)[0]

            # Obtenir l'indice de la classe avec la plus haute probabilité
            predicted_class_index = np.argmax(class_probabilities)

            # Mapper l'indice de la classe sur l'étiquette correspondante
            class_labels = ["T-shirt/top", "Pantalon", "Pull", "Robe", "Manteau", "Sandale", "Chemise", "Basket", "Sac", "Bottine"]
            predicted_class_label = class_labels[predicted_class_index]

            return jsonify({'product': predicted_class_label})

    return render_template("predict.html")
@app.route('/download', methods=['POST'])
def download():
    dataset_url = request.form.get('url')
    if dataset_url:
        try:
            response = requests.get(dataset_url)
            if response.status_code == 200:
                content = response.content
                return jsonify({'success': True, 'content': content.decode()})
            else:
                return jsonify({'success': False, 'error': 'Failed to download the file.'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'URL not provided.'})


@app.route('/displayURL', methods=['POST'])
def displayURL():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        num_rows = int(request.form['num_rows'])
        num_cols = int(request.form['num_cols'])
        filename = request.form['filename']  # Nom du fichier à traiter

        
        # Pour cet exemple, je vais créer un jeu de données fictif
        df_train = []
        for i in range(num_rows):
            row = {}
            for j in range(num_cols):
                row[f'Col{j+1}'] = f'Data{i+1}-{j+1}'
            df_train.append(row)

        # Ensuite, vous pouvez renvoyer les données au format JSON
        return jsonify(df_train)


if __name__ == '__main__':
    app.run(debug=True)

