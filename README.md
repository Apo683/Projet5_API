# Projet5_API

Description : API /predict_tags/ qui prend en entrée une question de StackoverFlow sous format json avec un 'Title' et 'Body' et renvoie les tags associés via notre modèle de prédiction.


Solution sélectionnée : TF-IDF pour la vectorisation (choisi car il demande moins de ressources que d'autres alternatives) et SVC (Support Vector Classifier) pour la classification multilabel et la prédiction des tags.


Prérequis : instance avec au moins 4Go de RAM (pour le chargement des modèles et l'exécution des prédictions).


Découpage des dossiers de l'API :


-  config.py pour les variables d'environnement de l'API, y compris les chemins vers les modèles entrainés nécessaires à la solution ainsi que le threshold (seuil à partir du quel un tag est associé à la question).

-  utils.py regroupe des fonctions utilitaires pour le bon fonctionnement de l'API (nettoyage et prétraitement du texte, vectorisation).

-  main.py : fichier principal qui configure et lance l'API FastAPI.

-  ml_model.py : gère le chargement des modèles et l'inférence (prédiction sur de nouvelles données).

-  models.py : définit la structure des données d'entrée attendues par l'API (schemas).

-  /routers/predict.py : contient le cœur de la solution. Il reçoit les requêtes, nettoie et prétraite le texte, le vectorise, puis utilise le modèle SVC pour prédire les tags correspondants.

-  /scripts/download_models.py : télécharge localement les modèles stockés sur S3

-  /models/ : dossier qui contient l'ensemble des modèles entrainés après téléchargement depuis S3 grâce au script download_models.py

-  /resources/.env : fichier à créer manuellement qui contient les credentials AWS pour le téléchargement des modèles sur S3


-  ~/tests/ : contient les scripts pour les tests unitaires à exécuter avec pytest

-  ~/requirements.txt : fichier listant les packages nécessaires à installer avec pip
