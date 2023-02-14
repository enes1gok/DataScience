import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


data = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/MushroomProject/mushrooms.csv")
#print(data.head())

classes = data['class'].value_counts()
#print(classes) #edible or poisonous

'''plt.bar('Edible', classes['e'])
plt.bar('Poisonous', classes['p'])
plt.show()'''

x = data.loc[:,['cap-shape', 'cap-color', 'ring-number', 'ring-type']]
y = data.loc[:,'class']

encoder = LabelEncoder() #convert to integer values
for i in x.columns:
    x[i] = encoder.fit_transform(x[i])

y = encoder.fit_transform(y)
'''print(x)
print(y)'''
x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #split the data into train and test with 70-30 ratio

#COMPARING OUR MODELS
#WE NEED TO CHOOSE BEST MODEL SO WE SHOULD LOOK ALL OF THEM FROM OUR CHOICES.

logistic_classifier_model = LogisticRegression()

ridge_classifier_model = RidgeClassifier()

decision_tree_model = DecisionTreeClassifier()

naive_bayes_model = GaussianNB()

neural_network_model = MLPClassifier()

#TRAIN ALL MODELS USING .fit() METHOD OF EACH OBJECT

logistic_classifier_model.fit(x_train, y_train)

ridge_classifier_model.fit(x_train, y_train)

decision_tree_model.fit(x_train, y_train)

naive_bayes_model.fit(x_train, y_train)

neural_network_model.fit(x_train, y_train)

#USING THE X_TEST SET WE MAKE PREDICTIONS WITH EACH MODEL AND SAVE RESULTS TO CORRESPONDING VARIABLE

logistic_classifier_pred = logistic_classifier_model.predict(x_test)

ridge_classifier_pred = ridge_classifier_model.predict(x_test)

decision_tree_pred = decision_tree_model.predict(x_test)

naive_bayes_pred = naive_bayes_model.predict(x_test)

neural_network_pred = neural_network_model.predict(x_test)

#COMPARING PERFORMANCES

logistic_report = classification_report(y_test, logistic_classifier_pred)
print("**********LOGISTIC REGRESSION**********")
print(logistic_report)
ridge_report = classification_report(y_test, ridge_classifier_pred)
print("**********RIDGE REGRESSION**********")
print(ridge_report)
decision_tree_report = classification_report(y_test, decision_tree_pred)
print("**********DECISION TREE**********")
print(decision_tree_report)
naive_bayes_report = classification_report(y_test, naive_bayes_pred)
print("**********NAIVE BAYES**********")
print(naive_bayes_report)
neural_network_report = classification_report(y_test, neural_network_pred)
print("**********NEURAL NETWORK**********")
print(neural_network_report)

#DECISION TREE PERFORMED BEST LET'S LOOK ONE STEP FURTHER AND TRY THE RANDOM FOREST ALGORITHM TO SEE IF IT WORKS BETTER

random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train, y_train)
random_forest_pred = random_forest_model.predict(x_test)
random_forest_report = classification_report(y_test, random_forest_pred)
print("**********RANDOM FOREST**********")
print(random_forest_report)

#IT DOESN'T WORK!!!!