import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

"""
Importation du dataset
"""
dataset = pd.read_csv("UniversalBank.csv")

"""
dataset.info()
"""

"""
On fractionne le dataset en deux ensembles
Un ensemble d'apprentissage et un ensemble de test
"""
train_set, test_set = train_test_split(dataset, test_size = 0.26, random_state = None) 
# 1300 données seront utilisées pour les test et donc 3700 données pour l'apprentissage

"""
print("Info sur les données d'apprentissage : \n")
train_set.info()
print("\n\nInfo sur les données de test : \n")
test_set.info()
"""

"""
On classifie a partir des données d'apprentissage et test
les prédicteurs (les variables indépendantes X) et
la cible (la variable dépendante Y)
"""
#Les données d'entrainement
X_train = train_set.iloc[:, [0,1,2,3,4,5,6,7,8,10,11,12,13]].values
Y_train = train_set.iloc[:, 9].values

#Les données de test
X_test = test_set.iloc[:, [0,1,2,3,4,5,6,7,8,10,11,12,13]].values
Y_test = test_set.iloc[:, 9].values

# Mise à l'échelle des données (Normalisation)
standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

"""
Création du modèle d'apprentisage
"""
classifier = SVC(kernel = 'rbf', random_state = None)
classifier.fit(X_train, Y_train) #Entrainement du modèle

#Affichage du nombre de vecteurs de support par classe
NSupport = classifier.n_support_
print(NSupport[0], "vecteurs de support pour la  classe 0 (Ceux qui refusent la proposition de prêt) et", NSupport[1], "vecteurs de support pour la classe 1 (Ceux qui acceptent la proposition de prêt)\n")
#print(classifier.support_vectors_)


Y_pred = classifier.predict(X_test) #Prediction de la cible Y_pred en fonction des variables indépendantes de test

"""
# Affichage de la variable dépendante réelle de tests et
# la variable dépendante prédicte à partir des variables indépendantes de test
print("Y réelles\tY prédictes\n")
for i in range(len(Y_test)) :
    print(Y_test[i], "\t", Y_pred[i])
"""

#Matrice de confusion 
print("\n\nLa matrice de confusion est : \n", confusion_matrix(Y_test, Y_pred), "\n")

accuracy = float(confusion_matrix(Y_test, Y_pred).diagonal().sum())/len(Y_test)
#Précision du modèle d'apprentisage
print("La précision de ce modèle est donc estimé à : ", accuracy, "\n", (1-accuracy)*100, "% d'erreur comise par la machine à vecteur de support")

