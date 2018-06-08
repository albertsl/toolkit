#Imports for data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def main():
	##########
	# # General Data Science
	##########

	# # First step: Visualizing the data
	# file = 'data.csv'
	# df = pd.read_csv(file)
	# df.head()
	# df.info()
	# df.describe()
	# df.columns
	# # Second step: Exploratory Data Analysis (EDA)
	# sns.pairplot(df)
	# sns.distplot(df['column'])
	# sns.heatmap(df.corr())
	# # Third step: Preparing the data
	# x = df[['column1','column2','column3','etc']] # The list of columns equals df.columns minus the column we want to predict and non numeric columns.
	# y = df['column to predict']
	# # Fourth step: Split data into a train / test dataset
	# from sklearn.model_selection import train_test_split
	# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 101)

	##########
	# # Linear Regression
	##########

	# # Fifth step: Train linear regression model
	# from sklearn.linear_model import LinearRegression
	# lm = LinearRegression()
	# lm.fit(x_train,y_train)
	# # Sixth step: Linear model interpretation
	# lm.intercept_
	# lm.coef_
	# # Seven: Use the model to predict
	# predictions = lm.predict(x_test)
	# # Eight: Evaluate the accuracy of the model
	# plt.scatter(y_test,predictions) #should have the shape of a line for a good predictions
	# sns.distplot(y_test-predictions) #should be a normal distribution centered in 0
	# from sklearn import metrics
	# metrics.mean_absolute_error(y_test, predictions)
	# metrics.mean_squared_error(y_test, predictions)
	# np.sqrt(metrics.mean_squared_error(y_test, predictions))
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # Logistic Regression
	##########

	# # Fifth step: Train logistic regression model
	# from sklearn.linear_model import LogisticRegression
	# logmodel = LogisticRegression()
	# logmodel.fit(x_train,y_train)
	# # Sixth step: Evaluate the model
	# predictions = logmodel.predict(x_test)
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # KNN
	##########

	# # Third step: Standarizing data
	# from sklearn.preprocessing import StandardScaler
	# scaler = StandardScaler()
	# scaler.fit(df.drop('TARGET CLASS', axis = 1))
	# scaled_features = scaler.transform(df.drop())
	# df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
	# # Fifth step: Train KNN model
	# from sklearn.neighbors import KNeighborsClassifier
	# knn = KNeighborsClassifier(n_neighbors = 1)
	# knn.fit(x_train, y_train)
	# # Sixth step: Evaluate the model
	# predictions = knn.predict(x_test)
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # Decision Tree
	##########

	# # Fifth step: Train Decision Tree model
	# from sklearn.tree import DecisionTreeClassifier
	# dtree = DecisionTreeClassifier()
	# dtree.fit(x_train, y_train)
	# # Sixth step: Evaluate the model
	# predictions = dtree.predict(x_test)
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # Random Forests
	##########

	# # Fifth step: Train Random Forests model
	# from sklearn.ensemble import RandomForestClassifier
	# rfc = RandomForestClassifier(n_estimators=200)
	# rfc.fit(x_train, y_train)
	# # Sixth step: Evaluate the model
	# predictions = rfc.predict(x_test)
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # Support Vector Machine (SVM)
	##########

	# # Fifth step: Train SVM model
	# from sklearn.svm import SVC
	# model = SVC()
	# model.fit(x_train, y_train)
	# # If it's not working well, refine parameters
	# from sklearn.grid_search import GridSearchCV
	# param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
	# grid = GridSearchCV(SVC(), param_grid, verbose = 3)
	# grid.best_params_
	# grid.best_estimator_
	# model.fit(x_train, y_train) #Add new found parameters
	# # Sixth step: Evaluate the model
	# predictions = rfc.predict(x_test)
	# from sklearn.metrics import classification_report
	# print(classification_report(y_test,predictions))
	# from sklearn.metrics import confusion_matrix
	# print(confusion_matrix(y_test,predictions))

	##########
	# # K-Means Clustering
	##########

	# # First step: Train model
	# from sklearn.cluster import KMeans
	# kmeans = KMeans(n_clusters=K) #Choose K
	# kmeans.fit(data)
	# # Second step: Evaluate the model
	# kmeans.cluster_centers_
	# kmeans.labels_

if __name__ == "__main__":
	main()
