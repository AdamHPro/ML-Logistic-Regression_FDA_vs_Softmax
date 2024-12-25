import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from predict import predict
from sklearn.preprocessing import StandardScaler


# Load data (Wine dataset)
np.random.seed(1)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')
np.random.shuffle(my_data)  # shuffle datataset

n_train = 100
X_train = my_data[:n_train, 1:]  # training data
y_train = my_data[:n_train, 0]  # class labels of training data

X_test = my_data[n_train:, 1:]  # training data
y_test = my_data[n_train:, 0]  # class labels of training data

# Initialiser et entraîner un modèle de régression logistique multinomiale
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Les paramètres θ (coef et intercept)
theta = model.coef_  # Coefficients pour chaque classe
intercept = model.intercept_  # Biais pour chaque classe
print("Paramètres θ (coefficients) :\n", theta)
print("Paramètres biais (intercept) :\n", intercept)

# Prédire les labels sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nPrécision (Accuracy) sur les données de test : {:.2f}%".format(accuracy * 100))

