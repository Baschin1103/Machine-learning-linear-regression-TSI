### Script for the multiple linear regression model ###
## The aim is to create a model to forecast the Total-Skill-Index (TSI) of a player ##
# https://wiki.hattrick.org/wiki/Total_Skill_Index #


# Import of nessesary libraries

import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import numpy as np



# Loading the prepared data (players are not older than 30 years old and without goalkeepers)

df = pd.read_csv('transferresultsplayers_1.csv', sep=',') #Careful: The csv's have a problem with the collumn 'Verletzungen ': the empty-space needs to be deleted 
df2 = pd.read_csv('transferresultsplayers_2.csv', sep=',')
df3 = pd.read_csv('transferresultsplayers_3.csv', sep=',')
df4 = pd.read_csv('transferresultsplayers_4.csv', sep=',')
df5 = pd.read_csv('transferresultsplayers_5.csv', sep=',')
df6 = pd.read_csv('transferresultsplayers_6.csv', sep=',')
df7 = pd.read_csv('transferresultsplayers_7.csv', sep=',')

frames = [df, df2, df3, df4, df5, df6, df7]

filled_data = pd.concat(frames, ignore_index=True)
df = filled_data

df = df.filter(['Alter', 'Form', 'Kondition', 'Verteidigung', 'Spielaufbau', 'Flügelspiel', 'Passspiel', 'Torschuss', 'TSI'], axis=1)



# The csv of my own players is in a differently formatted table

meine_df = pd.read_csv('players_own.csv', sep=',')

meine_df = meine_df.filter(['Alter', 'Form', 'Kondition', 'Verteidigung', 'Spielaufbau', 'Flügelspiel', 'Passspiel', 'Torschuss', 'TSI'], axis=1)

frames = [df, meine_df]

filled_data = pd.concat(frames, ignore_index=True)
df = filled_data

print('Dataframe:\n', df)

# Export of dataframe for example: df.to_csv('all_players.csv', encoding='utf-8', index=False, header=True)



# Looking for missing values in collumns

print('Missing values:\n', df.isnull().sum())




# Defining the dependend variable

y = df['TSI']

print('y:\n', y)


# Defining the independent variables 

df = df.drop('TSI', axis = 1)

X = df
print('X:\n', X)




# Data-splitting for training and testing

from sklearn.model_selection import train_test_split   # import of script to split data for training- and testingdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)    # 70% of the data for training, 30% für testing

X_train = X
y_train = y





# Training of the multiple linear regression model

from sklearn.linear_model import LinearRegression   # import of script with linear regression model
lr = LinearRegression()     

lr.fit(X_train, y_train)    # Training the model with the trainingdata




# Results for the regression-model

c = lr.intercept_       # estimated intercept of dependend variable
print('c:', round(c, 3))

m = lr.coef_            #  estimated slopes/relationship between the independend varibales and dependend variable
np.set_printoptions(precision=3) 
print('m:', m)





### Quality of results ###

y_pred_train = lr.predict(X_train)  # predicted y from the trained model with the training-data

y_pred_test = lr.predict(X_test)    # predicted y from the trained model with the test-data



## Plotting of the prediction of y and the actual y of the datasets ##
# Plotting of the prediction of y (training data) and the actual y of the training dataset

plt.subplot(1,2,1)

plt.scatter(y_train,y_pred_train)
plt.xlabel("Actual TSI")        
plt.ylabel("Predicted TSI")
plt.title("Training data")
plt.grid()


# Looking for overfitting by the testing data
# Plotting of the prediction of y (testing data) and the actual y of the testing dataset

plt.subplot(1,2,2)

plt.scatter(y_test,y_pred_test, color='r')
plt.xlabel("Actual TSI")
plt.ylabel("Predicted TSI")
plt.title("Testing data")
plt.grid()

plt.show()



## Accuracy of results ##
# r^2 value for the training data

from sklearn.metrics import r2_score
r2 = r2_score(y_train, y_pred_train)        # Quality criterion for the training data: Variance-Explanation of y in percent
print('r2_train:', round(r2, 4))


# r^2 value for the testing data

r2 = r2_score(y_test, y_pred_test)          # Quality criterion for the testing data: Variance-Explanation of y in percent
print('r2_test:', round(r2, 4))


   

## Presentation of test-results as numbers ##
# Creating a numpy-matrix for calculating

m = np.zeros((48,3), dtype=float)
m[:, 0] = y_test
m[:, 1] = y_pred_test


# Calculation of the errors between actual TSI and predictions

error = (np.subtract(y_test, y_pred_test))
m[:, 2] = error

#np.set_printoptions(suppress=True) # Supressing scientific view of numbers if needed



# Creating a dataframe for better viewing

DF = pd.DataFrame(m, columns=['Actual TSI','Prediction','Error']) 
DF = DF.round(decimals=2)
DF = DF.to_string(index=False)
print('Results for the testing data:\n', DF)


#Export of results as csv for example: DF.to_csv('results_lin_reg_TSI.csv', encoding='utf-8', index=False)