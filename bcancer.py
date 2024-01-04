# Exercice Machine Learning : 

# 1. Créer un fichier python bcancer.py qui va entrainer un modèle de ML pour prédire la présence de cellules cancéreuses : 

##############
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Charger les données
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convertir en DataFrame pour une meilleure manipulation
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Diviser les données en ensembles d'entraînement et de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)


# Évaluer le modèle

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Sauvegarder le modèle entraîné au format pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modèle entraîné et sauvegardé.")
###############

#2. Créer un fichier python test.py qui va tester la qualité de prédiction du modèle sur un nouvel échantillon de données (disponible ici https://github.com/AbdallahTayeb/DevOps-Course/blob/main/sample.csv) : 


import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Charger ou préparer les données de test
df = pd.read_csv('sample.csv',sep=';')
X_test = df.drop('target', axis=1)
y_test = df['target']

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Définir un seuil de classification
seuil_classification = 0.90  # par exemple

# Vérifier si le seuil est atteint
if accuracy >= seuil_classification:
    print("Le seuil de classification est atteint. Le modèle est prêt pour le déploiement.")
else:
    print("Le seuil de classification n'est pas atteint. Le modèle nécessite une amélioration.")


# 3. Créer un workflow avec un job build qui va executer bcancer.py et un job test qui va executer test.py
