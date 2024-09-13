import os
import boto3
from botocore.exceptions import ClientError
import joblib

# Configuration du bucket S3 et des fichiers à télécharger
s3 = boto3.client('s3')
bucket_name = 'sagemaker-eu-west-3-741448955370'

# Fichiers à télécharger
files_to_download = {
    'mlb_fit.pkl': 'joblib/mlb_fit.pkl',
    'svc_model_BERT_model.pkl': 'mlruns/svc_model_BERT/model.pkl'
}

# Dossier local où les fichiers seront stockés
local_model_directory = '/app/models/'

# Fonction pour télécharger un fichier depuis S3
def download_from_s3(s3_key, local_path):
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Fichier {s3_key} téléchargé avec succès dans {local_path}")
    except ClientError as e:
        print(f"Erreur lors du téléchargement de {s3_key} : {e}")

# Télécharger les fichiers si ils n'existent pas déjà localement
if not os.path.exists(local_model_directory):
    os.makedirs(local_model_directory)

for local_file_name, s3_key in files_to_download.items():
    local_path = os.path.join(local_model_directory, local_file_name)
    if not os.path.exists(local_path):
        print(f"Téléchargement de {s3_key} depuis S3...")
        download_from_s3(s3_key, local_path)
    else:
        print(f"Le fichier {local_file_name} existe déjà localement, téléchargement ignoré.")

# Charger les modèles après les avoir téléchargés
mlb = joblib.load(os.path.join(local_model_directory, 'mlb_fit.pkl'))
svc_model = joblib.load(os.path.join(local_model_directory, 'svc_model_BERT_model.pkl'))

# Vous pouvez maintenant utiliser ces modèles dans votre API