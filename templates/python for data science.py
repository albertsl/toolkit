#Albert Sanchez Lafuente 2/4/2019, Pineda de Mar, Spain
#https://github.com/albertsl/
#Structure of the template mostly based on the Appendix B of the book Hands-on Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron (https://amzn.to/2WIfsmk)
#Big thank you to Uxue Lazcano (https://github.com/uxuelazkano) for code on model comparison
#Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
	#Load data
	df = pd.read_csv('file.csv')
	#If data is too big, take a sample of it
	df = pd.read_csv('file.csv', nrows=50000)

	#Visualize data
	df.head()
	df.describe()
	df.info()
	df.columns
	#For a categorical dataset we want to see how many instances of each category there are
	df['categorical_var'].value_counts()

	#Exploratory Data Analysis (EDA)
	sns.pairplot(df)
	sns.distplot(df['column'])
	sns.countplot(df['column'])
	
	#Fix or remove outliers
	plt.boxplot(df['feature1'])
	plt.boxplot(df['feature2'])

	#Check for missing data
	total_null = df.isna().sum().sort_values(ascending=False)
	percent = (df.isna().sum()/df.isna().count()).sort_values(ascending=False)
	missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
	#Generate new features with missing data
	df['feature1_nan'] = df['feature1'].isna()
	df['feature2_nan'] = df['feature2'].isna()

	#Check for duplicated data
	df.duplicated().value_counts()
	df['duplicated'] = df.duplicated() #Create a new feature

	#Fill missing data or drop columns/rows
	df.fillna()
	df.drop('column_full_of_nans')
	df.dropna(how='any')

	#Correlation analysis
	sns.heatmap(df.corr(), annot=True, fmt='.2f')

	#Feature selection: Drop attributes that provide no useful information for the task

	#Feature engineering. Create new features by transforming the data
	#Discretize continuous features
	#Decompose features (categorical, date/time, etc.)
	#Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.)
	#Aggregate features into promising new features (x*y)
	#For time series data
	from astropy.stats import median_absolute_deviation
	from statsmodels.robust.scale import mad
	for column in df.columns:
		df[column + '_mean'] = df.groupby(['series_id'])[column].mean()
		df[column + '_median'] = df.groupby(['series_id'])[column].median()
		df[column + '_max'] = df.groupby(['series_id'])[column].max()
		df[column + '_min'] = df.groupby(['series_id'])[column].min()
		df[column + '_std'] = df.groupby(['series_id'])[column].std()
		df[column + '_range'] = df[column + '_max'] - df[column + '_min']
		df[column + '_max_over_Min'] = df[column + '_max'] / df[column + '_min']
		df[column + 'median_abs_dev'] = df.groupby(['series_id'])[column].mad()
	#For speed/movement data, add vectorial features. Try many different combinations
	df['position_norm'] = df['position_X'] ** 2 + df['position_Y'] ** 2 + df['position_Z'] ** 2
    df['position_module'] = df['position_norm'] ** 0.5
    df['position_norm_X'] = df['position_X'] / df['position_module']
    df['position_norm_Y'] = df['position_Y'] / df['position_module']
    df['position_norm_Z'] = df['position_Z'] / df['position_module']
	df['position_over_velocity'] = data['position_module'] / data['velocity_module']
	#Create a new column from conditions on other columns
	df['column_y'] = df[(df['column_x1'] | 'column_x2') & 'column_x3']
	df['column_y'] = df['column_y'].apply(bool)
	df['column_y'] = df['column_y'].apply(int)
	#Create a new True/False column according to the first letter on another column.
	lEI = [0] * df.shape[0]

	for i, row in df.iterrows():
		try:
			l = df['room_list'].iloc[i].split(', ')
		except:
			#When the given row is empty
			l = []
		for element in l:
			if element[0] == 'E' or element[0] == 'I':
				lEI[i] = 1

	df['EI'] = pd.Series(lEI)

	#Scaling features
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	scaler.fit(df)
	df_norm = scaler.transform(df)
	
	#Define Validation method
	#Train and validation set split
	from sklearn.model_selection import train_test_split
	X = df.drop('target_var', inplace=True, axis=1)
	y = df['column to predict']
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, random_state = 101)
	#Cross validation
	from sklearn.model_selection import cross_val_score
	cross_val_score(model, X, y, cv=5)
	#StratifiedKFold
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=5, random_state=101)
	for train_index, val_index in skf.split(X, y):
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]

	#Define Performance Metrics
	#ROC AUC for classification tasks
	from sklearn.metrics import roc_auc_score
	from sklearn.metrics import roc_curve
	roc_auc = roc_auc_score(y_val, model.predict(X_val))
	fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Model (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()
	#Confusion Matrix
	from sklearn.metrics import confusion_matrix
	confusion_matrix(y_val, y_pred)
	#MAE, MSE, RMSE
	from sklearn import metrics
	metrics.mean_absolute_error(y_val, y_pred)
	metrics.mean_squared_error(y_val, y_pred)
	np.sqrt(metrics.mean_squared_error(y_val, y_pred))
	#Classification report
	from sklearn.metrics import classification_report
	classification_report(y_val,y_pred)

    #Train many quick and dirty models from different categories(e.g., linear, naive Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.
	#########
	# Linear Regression
	#########
	from sklearn.linear_model import LinearRegression
	lr = LinearRegression()
	lr.fit(X_train,y_train)

	#Linear model interpretation
	lr.intercept_
	lr.coef_

	#Use model to predict
	y_pred = lr.predict(X_val)

	#Evaluate accuracy of the model
	plt.scatter(y_val, y_pred) #should have the shape of a line for good predictions
	sns.distplot(y_val - y_pred) #should be a normal distribution centered at 0
	acc_lr = round(lr.score(X_val, y_val) * 100, 2)

	#########
	# Logistic Regression
	#########
	from sklearn.linear_model import LogisticRegression
	logmodel = LogisticRegression()
	logmodel.fit(X_train,y_train)

	#Use model to predict
	y_pred = logmodel.predict(X_val)
	
	#Evaluate accuracy of the model
	acc_log = round(logmodel.score(X_val, y_val) * 100, 2)

	#########
	# KNN
	#########
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors = 5)
	knn.fit(X_train, y_train)

	#Use model to predict
	y_pred = knn.predict(X_val)
	
	#Evaluate accuracy of the model
	acc_knn = round(knn.score(X_val, y_val) * 100, 2)

	#########
	# Decision Tree
	#########
	from sklearn.tree import DecisionTreeClassifier
	dtree = DecisionTreeClassifier()
	dtree.fit(X_train, y_train)

	#Use model to predict
	y_pred = dtree.predict(X_val)
	
	#Evaluate accuracy of the model
	acc_dtree = round(dtree.score(X_val, y_val) * 100, 2)

	#########
	# Random Forest
	#########
	from sklearn.ensemble import RandomForestClassifier
	rfc = RandomForestClassifier(n_estimators=200, random_state=101)
	rfc.fit(X_train, y_train)

	from sklearn.ensemble import RandomForestRegressor
	rfr = RandomForestRegressor(n_estimators=200, random_state=101)
	rfr.fit(X_train, y_train)

	#Use model to predict
	y_pred = rfr.predict(X_val)
	
	#Evaluate accuracy of the model
	acc_rf = round(rfr.score(X_val, y_val) * 100, 2)

	#Evaluate feature importance
	importances = rfr.feature_importances_
	std = np.std([importances for tree in random_forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	feature_importances = pd.DataFrame(random_forest.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X_train.shape[1]), importances[indices], yerr=std[indices], align="center")
	plt.xticks(range(X_train.shape[1]), indices)
	plt.xlim([-1, X_train.shape[1]])
	plt.show()

	#########
	# Support Vector Machine (SVM)
	#########
	from sklearn.svm import SVC
	model = SVC()
	model.fit(X_train, y_train)

	#Use model to predict
	y_pred = model.predict(X_val)
	
	#Evaluate accuracy of the model
	acc_svm = round(model.score(X_val, y_val) * 100, 2)

	#########
	# K-Means Clustering
	#########
	#Train model
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=K) #Choose K
	kmeans.fit(df)
	#Evaluate the model
	kmeans.cluster_centers_
	kmeans.labels_

    #Measure and compare their performance
	models = pd.DataFrame({
    'Model': ['Linear Regression', 'Support Vector Machine', 'KNN', 'Logistic Regression', 
              'Random Forest'],
    'Score': [acc_lr, acc_svm, acc_knn, acc_log, 
              acc_rf]})
	models.sort_values(by='Score', ascending=False)
    #Analyze the most significant variables for each algorithm.
	#Analyze the types of errors the models make.
	#What data would a human have used to avoid these errors?
	#Have a quick round of feature selection and engineering.
	#Have one or two more quick iterations of the five previous steps.
	#Short-list the top three to five most promising models, preferring models that make different types of errors.

	#Fine-tune the hyperparameters using cross-validation
	#Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?)
	#Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach
	from sklearn.grid_search import GridSearchCV
	param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
	grid = GridSearchCV(model, param_grid, verbose = 3)
	model.fit(X_train, y_train)
	grid.best_params_
	grid.best_estimator_
	
	#Try Ensemble methods. Combining your best models will often perform better than running them individually

	#Once you are confident about your final model, measure its performance on the test set to estimate the generalization error

if __name__ == "__main__":
	main()
