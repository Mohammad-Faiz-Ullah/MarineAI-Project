import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# loading dataset to pandas dataframe
sonar_data = pd.read_csv('C:\\Users\\Faiz\\Desktop\\Minor Project (6th sem)\\sonar data.csv', header=None)


# a=sonar_data.head()   printing the dataset (pheli 5 rows aa jaengi)
# print(a)


# no.of rows and col.  (rows->208  col->60)
# print(sonar_data.shape)


# statistical measures for dataset (count ,mean, min, max etc)
# print(sonar_data.describe())

# how many rock and mine examples are there  (m->111 r->97)
# print(sonar_data[60].value_counts())


# grouping data based on rock and mine
# print(sonar_data.groupby(60).mean())


# separating data and labels (data->numerical value  lables-> last col. representing R and M)
x= sonar_data.drop(columns=60,axis=1) #storing datas in x  # basically axis=0 represents rows and axis=1 represents columns
y= sonar_data[60]  # storing labels in y
# print(x)
# print(y)


# TRAINING & TEST DATA
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=2)

# print(x.shape, x_train.shape, x_test.shape)  #LR,SVM,RF&KNN-->> the dataset is now split into train & test data with value: train->145 , test->63  (original->208)

# print(x_train) #training data (numerical values 145 vali)
# print(y_train) #training label (labels R & M)

# Model Training (Logistic Regression)
model = LogisticRegression()

# Model Training (SVM)
classifier = svm.SVC(kernel='linear')

# Model Training (RandomForest)
rf = RandomForestClassifier()

# Model Training (KNN)
knn = KNeighborsClassifier(n_neighbors=3)


# training logisticreression model with training data
model.fit(x_train, y_train)  # training done

# training SVM model with training data
classifier.fit(x_train, y_train)  # training done

# training RandomForest model with training data
rf.fit(x_train, y_train)  # training done

# training RandomForest model with training data
knn.fit(x_train, y_train)  # training done


# MODEL EVALUATION

#accuracy on training data (For LogisticRegression)
x_train_prediction= model.predict(x_train)
train_data_acc_lr= accuracy_score(x_train_prediction,y_train)
print("Accuracy score for LogisticRegression (on training data) =", train_data_acc_lr) #83%

#accuracy on test data (For LogisticRegression)
x_test_prediction= model.predict(x_test)
test_data_acc_lr= accuracy_score(x_test_prediction,y_test)
print("Accuracy score for LogisticRegression (on test data) =",test_data_acc_lr,"\n") #76%

#CONFUSION MATRIX (For LogisticRegression)
ConfusionMatrixDisplay.from_estimator(model,x_test,y_test,display_labels=['Mine', 'Rock'])
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#accuracy on training data (For SVM)
xs_train_prediction = classifier.predict(x_train)
train_data_acc_svm= accuracy_score(xs_train_prediction,y_train)
print("Accuracy score for SVM (on training data) =", train_data_acc_svm) #82%

#accuracy on test data (For SVM)
xs_test_prediction= classifier.predict(x_test)
test_data_acc_svm= accuracy_score(xs_test_prediction,y_test)
print("Accuracy score for SVM (on test data) =",test_data_acc_svm,'\n') #85%

#CONFUSION MATRIX (For SVM)
ConfusionMatrixDisplay.from_estimator(classifier,x_test,y_test,display_labels=['Mine', 'Rock'])
plt.title("Confusion Matrix - SVM")
plt.show()

#accuracy on training data (For RandomForest)
xr_train_prediction = rf.predict(x_train)
train_data_acc_rf= accuracy_score(xr_train_prediction,y_train)
print("Accuracy score for RandomForest (on training data) =", train_data_acc_rf) #100%

#accuracy on test data (For RandomForest)
xr_test_prediction= rf.predict(x_test)
test_data_acc_rf= accuracy_score(xr_test_prediction,y_test)
print("Accuracy score for RandomForest (on test data) =",test_data_acc_rf,'\n') #84%

#CONFUSION MATRIX (For RF)
ConfusionMatrixDisplay.from_estimator(rf,x_test,y_test,display_labels=['Mine', 'Rock'])
plt.title("Confusion Matrix - RandomForest")
plt.show()

#accuracy on training data (For KNN)
xn_train_prediction = knn.predict(x_train)
train_data_acc_knn= accuracy_score(xn_train_prediction,y_train)
print("Accuracy score for KNN (on training data) =", train_data_acc_knn) #86%

#accuracy on test data (For KNN)
xn_test_prediction= knn.predict(x_test)
test_data_acc_knn= accuracy_score(xn_test_prediction,y_test)
print("Accuracy score for KNN (on test data) =",test_data_acc_knn,'\n') #70%

#CONFUSION MATRIX (For KNN)
ConfusionMatrixDisplay.from_estimator(knn,x_test,y_test,display_labels=['Mine', 'Rock'])
plt.title("Confusion Matrix - KNN")
plt.show()


# MAKING A PREDICTIVE SYSTEM 

input = (0.0516,0.0944,0.0622,0.0415,0.0995,0.2431,0.1777,0.2018,0.2611,0.1294,0.2646,0.2778,0.4432,0.3672,0.2035,0.2764,0.3252,0.1536,0.2784,0.3508,0.5187,0.7052,0.7143,0.6814,0.5100,0.5308,0.6131,0.8388,0.9031,0.8607,0.9656,0.9168,0.7132,0.6898,0.7310,0.4134,0.1580,0.1819,0.1381,0.2960,0.6935,0.8246,0.5351,0.4403,0.6448,0.6214,0.3016,0.1379,0.0364,0.0355,0.0456,0.0432,0.0274,0.0152,0.0120,0.0129,0.0020,0.0109,0.0074,0.0078)

input_numpy = np.asarray(input) #changing input data to numpy array

input_reshape = input_numpy.reshape(1,-1) #Reshaping np array, as we are predictating for one instance

prediction_1= model.predict(input_reshape)  #for LR
prediction_2= classifier.predict(input_reshape)  #for SVM
prediction_3= rf.predict(input_reshape)  #for RandomForest
prediction_4= knn.predict(input_reshape)  #for KNN

print("\nLogistic Regression Result:-")
print(prediction_1)
if(prediction_1[0]=='R'):    
    print("Object is Rock")

else:
    print("Object is Mine")

print("\nSVM Result:-")
print(prediction_2)
if(prediction_2[0]=='R'):    
    print("Object is Rock")

else:
    print("Object is Mine")

print("\nRandomForest Result:-")
print(prediction_3)
if(prediction_3[0]=='R'):    
    print("Object is Rock")

else:
    print("Object is Mine")

print("\nKNN Result:-")
print(prediction_4)
if(prediction_4[0]=='R'):    
    print("Object is Rock")

else:
    print("Object is Mine")


# instances=[(0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115)
#            ,(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
#            ]

# for i,instances in enumerate(instances):
#     input_num = np.asarray(instances)
#     input_reshape = input_num.reshape(1,-1)
#     prediction_lr = model.predict(input_reshape)
#     prediction_svm = classifier.predict(input_reshape)
#     prediction_rf = rf.predict(input_reshape)
#     prediction_knn = knn.predict(input_reshape)

# print(f"\n instances {i+1}:")
# print("linearegression predict", prediction_lr[0])

# print("svm predict", prediction_svm[0])

# print("randomforest predict", prediction_rf[0])

# print("KNN predict", prediction_knn[0])

# Graph Plotting
accuracy= [train_data_acc_lr, test_data_acc_lr, train_data_acc_svm, test_data_acc_svm,train_data_acc_rf,test_data_acc_rf,train_data_acc_knn,test_data_acc_knn]
labels = ['LR Train', 'LR Test', 'SVM Train', 'SVM Test','RF Train','RF Test','KNN Train','KNN Test']

plt.figure(figsize=(10,6))
plt.bar(labels,accuracy)
plt.title("Comparison graph b/w LR, SVM, RF & KNN")
plt.xlabel("Models & Datasets")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.grid(axis='y', linestyle='-', alpha=1)
plt.show()

# print("\nTesting Data:")
# print("X_test (features):")
# print(x_test)
# print("y_test (labels/targets):")
# print(y_test)