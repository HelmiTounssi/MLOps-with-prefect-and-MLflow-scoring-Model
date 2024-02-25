import os
from google.cloud import storage
import pandas as pd
import pickle
project_id = 'elevated-nuance-414716' 
bucket_name = 'data-model-home-credit' 
model_file = 'clf_lightgbm_100_fold_2_model_.pkl'
train_data_100 = 'processed_application_features_importances_100.csv'
test_data = 'test_data_final.pkl'
train_data = 'train_data_final.pkl'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elevated-nuance-414716-fec29a58b1f3.json'

# Initialise a client
client = storage.Client(project_id)
# Create a bucket object for our bucket
bucket = client.get_bucket(bucket_name)
# Create a blob object from the filepath
blob = bucket.blob(bucket_file)
# Upload the file to a destination
#blob.upload_from_filename(local_file)
# Note: Client.list_blobs requires at least package version 1.17.0.
blobs = client.list_blobs(bucket_name)

    # Note: The call returns a response only when the iterator is consumed.
for blob in blobs:
    print(blob.name)
    
blob = bucket.blob(train_data_100)  
df = pd.read_csv('gs://data-model-home-credit/processed_application_features_importances_100.csv')
print(df.head())
blob = bucket.blob(test_data) 
pickle_in = blob.download_as_string()
my_dictionary = pickle.loads(pickle_in)
print(my_dictionary.head())