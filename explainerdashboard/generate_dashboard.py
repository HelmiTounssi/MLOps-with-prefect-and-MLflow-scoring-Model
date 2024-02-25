import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

import sys
import os
# Obtenez le chemin absolu du répertoire parent du notebook
notebook_directory = os.path.abspath('')
# Construisez le chemin absolu du répertoire src
src_path = os.path.join(notebook_directory, 'home-credit-1.0.0/src/')
sys.path.insert(0, src_path)
# import data.features.fem_data  as fem
# import models.train.Boosting  as boost
# load a source module from a file
from google.cloud import storage
import pandas as pd
import pickle
import gcsfs
project_id = 'elevated-nuance-414716' 
bucket_name = 'data-model-home-credit' 
model_file = 'clf_lightgbm_100_fold_2_model_.pkl'
train_data_100 = 'processed_application_features_importances_100.csv'
test_data_path = 'test_data_final.pkl'
train_data = 'train_data_final.pkl'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elevated-nuance-414716-fec29a58b1f3.json'
# Initialise a client
client = storage.Client(project_id)
# Create a bucket object for our bucket
bucket = client.get_bucket(bucket_name)

train_data = pd.read_csv('gs://data-model-home-credit/processed_application_features_importances_100.csv')
fs = gcsfs.GCSFileSystem()
with fs.open('gs://data-model-home-credit/train_data_final.pkl', 'rb') as f:
    df = pickle.load(f)

'''blob1 = bucket.blob(train_data)
pickle_in = blob1.download_as_bytes()
df = pickle.loads(pickle_in)'''


#########################################################
#train_data=pd.read_csv('processed_application_features_importances_100.csv') 
#df = fem.read_pickle('train_data_final.pkl')



y_train = df['TARGET']
train_data = train_data.drop(['target'], axis=1)
train_data = train_data.drop(['prediction'], axis=1)
#test_data = fem.read_pickle('test_data_final.pkl')
with fs.open('gs://data-model-home-credit/test_data_final.pkl', 'rb') as f:
    test_data = pickle.load(f)
# Convert the set to a list
columns_list = list(train_data.columns)
# Now use the list to select the columns from the DataFrame
df_selected = test_data[columns_list]
test_data=df_selected

model=None
# 1. Charger le modèle à partir du fichier pickle
'''with open('./clf_lightgbm_100_fold_2_model_.pkl', 'rb') as f:
  model = pickle.load(f) '''
with fs.open('gs://data-model-home-credit/clf_lightgbm_100_fold_2_model_.pkl', 'rb') as f:
    model = pickle.load(f)
        
train_data_first_100 = train_data.iloc[:1000]
y_train_first_100=y_train.iloc[:1000]
explainer = ClassifierExplainer(model, train_data_first_100, y_train_first_100, 
                                # adds a table and hover labels to dashboard
                                labels=['Not Paid', 'Paid'], # defaults to ['0', '1', etc]
                                target = "Not Paid", # defaults to y.name
                                )

db = ExplainerDashboard(explainer, 
                        title="Home Credit Prediction", 
                        model_output='logodds',# defaults to "Model Explainer"
                        shap_interaction=False, # you can switch off tabs with bools
                        )

db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)