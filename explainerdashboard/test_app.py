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
import pickle
import sys
import os
# Obtenez le chemin absolu du répertoire parent du notebook
notebook_directory = os.path.abspath('')
src_path = os.path.join(notebook_directory, '../src/')

sys.path.insert(0, src_path)
import data.features.fem_data  as fem
import models.train.Modelling  as mod
import models.train.Boosting  as boost
# load a source module from a file


train_data=pd.read_csv('../data/final/processed_application_features_importances_100.csv') 
df = fem.read_pickle('../data/final/train_data_final.pkl')
y_train = df['TARGET']
train_data = train_data.drop(['target'], axis=1)
train_data = train_data.drop(['prediction'], axis=1)
test_data = fem.read_pickle('../data/final/test_data_final.pkl')
# Convert the set to a list
columns_list = list(train_data.columns)
# Now use the list to select the columns from the DataFrame
df_selected = test_data[columns_list]
test_data=df_selected
model=None
# 1. Charger le modèle à partir du fichier pickle
with open('./clf_lightgbm_100_fold_2_model_.pkl', 'rb') as f:
  model = pickle.load(f)
        
            #data['prediction_probs'] = model.predict_proba(X)[:, 1]
       

train_data_first_100 = train_data.iloc[:1000]
y_train_first_100=y_train.iloc[:1000]
explainer = ClassifierExplainer(model, train_data_first_100, y_train_first_100, 
                                # adds a table and hover labels to dashboard
                                model_output='logodds',
                                labels=['Not Paid', 'Paid'], # defaults to ['0', '1', etc]
                                target = "Not Paid", # defaults to y.name
                                )

db = ExplainerDashboard(explainer, 
                        title="Home Credit", # defaults to "Model Explainer"
                        shap_interaction=False, # you can switch off tabs with bools
                        )
db.run(port=8050)