#Albert Sanchez Lafuente - TodoIA
#https://github.com/albertsl/
#https://todoia.es/
#Structure of the template mostly based on the Appendix B of the book Hands-on Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron
#Load packages
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
#from tqdm.notebook import tqdm
import numba
np.random.seed(101)

#Check versions
import platform
print("Operating system:", platform.system(), platform.release())
import sys
print("Python version:", sys.version)
print("Numpy version:", np.version.version)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Numba version:", numba.__version__)
# Only Python 3.8:
from importlib import metadata
print('f{metadata.version("pandas")=}')
print('f{metadata.version("numpy")=}')
print('f{metadata.version("sklearn")=}')
print('f{metadata.version("matplotlib")=}')
print('f{metadata.version("seaborn")=}')
print('f{metadata.version("numba")=}')

#Only Jupyter notebook:
%matplotlib inline
%config InlineBackend.figure_format = 'svg' #or 'retina'. Makes the images more clear

#Manage dependencies with pipenv
#install pip env with: pip install --user pipenv 	Follow the online documentation for installing if there's any error, specially with Windows.
#Activate virtual environment
pipenv shell
pipenv install package
pipenv lock
#Run a command inside the virtual environment
pipenv run xxxxx
#Install dependencies in another computer
pipenv install --ignore-pipfile

#Logging: create a log file and write all messages there. Old messages are not deleted so log file may get very big. Interesting to add timedata to the beginning of each message.
import logging
logging.basicConfig(filename='log_file.log', level=logging.INFO) #Will show all logging with same or more importance as INFO
logging.debug('message')
logging.info('message')
logging.warning('message')
logging.error('message')
logging.critical('message')

#Load data
df = pd.read_csv('file.csv', sep=',', skiprows=0)
df = pd.read_excel('file.xlsx')
#If data is too big, take a sample of it
df = pd.read_csv('file.csv', nrows=50000)
#Load mat file
from scipy.io import loadmat
data = loadmat('file.mat')

#Get all files from a folder
from os import listdir
from os import path
main_folders = ['f1', 'f2']
for folder in main_folders:
	lfolder = listdir(folder)
	for f in lfolder:
		path.join(folder, f)

#Get memory usage
df.memory_usage().sum() / 1024**2 #MB

#Reduce dataframe memory usage
def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
		to reduce memory usage.
	"""
	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
	
	for col in df.columns:
		col_type = df[col].dtype
		
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')

	end_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	
	return df
#Sometimes it changes some values in the dataframe, let's check it doesn't change anything
df_test = pd.DataFrame()
df_opt = reduce_mem_usage(df)
for col in df:
	df_test[col] = df[col] - df_opt[col]
#Mean, max and min for all columns should be 0
df_test.describe().loc['mean']
df_test.describe().loc['max']
df_test.describe().loc['min']

#Save dataframe as csv
df.to_csv('data.csv', index=False)

#Save dataframe as h5
df.to_hdf('df.h5', key='df', mode='w')
#Read dataframe from h5
df = pd.read_hdf('df.h5', 'df')

#Improve execution speed of your code by adding these decorators:
@numba.jit
def f(x):
	return x
@numba.njit #The nopython=True option requires that the function be fully compiled (so that the Python interpreter calls are completely removed), otherwise an exception is raised.  These exceptions usually indicate places in the function that need to be modified in order to achieve better-than-Python performance.  We strongly recommend always using nopython=True.
def f(x):
	return x

#Speed-up pandas apply time:
import swifter
df.swifter.apply(lambda x: x.sum() - x.min())

#Speed-up pandas using GPU with cudf. Install using:
#conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf
import cudf
cudf_df = cudf.DataFrame.from_pandas(df)
#we can use all the same methods as we use in pandas
cudf_df['col1'].mean()
cudf_df.merge(cudf_df2, on='b')

#Error management:
try:
    x*3
except Exception as e:
    print(type(e))
    print(e.args)
    print(e)

#Styling pandas DataFrame visualization https://pbpython.com/styling-pandas.html
#https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
# more info on string formatting: https://mkaz.blog/code/python-string-format-cookbook/
format_dict = {'price': '${0:,.2f}', 'date': '{:%m-%Y}', 'pct_of_total': '{:.2%}'}
#Format the numbers
df.head().style.format(format_dict).hide_index()
#Highlight max and min
df.head().style.format(format_dict).hide_index().highlight_max(color='lightgreen').highlight_min(color='#cd4f39')
#Colour gradient in the background
df.head().style.format(format_dict).background_gradient(subset=['sum'], cmap='BuGn')
#Bars indicating number size
df.head().style.format(format_dict).hide_index().bar(color='#FFA07A', vmin=100_000, subset=['sum'], align='zero').bar(color='lightgreen', vmin=0, subset=['pct_of_total'], align='zero').set_caption('2018 Sales Performance')

#Visualize data
df.head()
df.describe()
df.info()
df.columns
df.nunique()
df.unique()
#For a categorical dataset we want to see how many instances of each category there are
df['categorical_var'].value_counts()
#Automated data visualization
from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
#For a big dataset that takes too long to create the report, use minimal. It will generate a simplified report.
profile = ProfileReport(df, minimal=True)
profile.to_file(output_file="output_min.html")
#Interactive data visualization
import dtale
d = dtale.show(df)
d.open_browser()

#Add rows from Series to a DataFrame
s = pd.Series([data], index=df.columns)
df = df.append(s, ignore_index=True)

#Define a pipeline
from sklearn.pipeline import pipeline
prepare_and_train = Pipeline([()'scaler', MinMaxScaler()), ('svm', SVC())])
prepare_and_train.fit(X_train, y_train)
prepare_and_train.score(X_val, y_val)
#Pipeline with Cross-validation
feat_selection_and_train = Pipeline([("select", SelectPercentile(percentile=5)), ("ridge", Ridge())])
np.mean(cross_val_score(feat_selection_and_train, X, y, cv=5))
#Pipeline for Grid Search
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
			  'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipe, param_grid=param_grid, cv=5)
gs.fit(X_train, y_train)

gs.best_score_
gs.score(X_val, y_val)
gs.best_params_
#Applying different transformations to different columns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
ohe = OneHotEncoder()
si = SimpleImputer()
ct = make_column_transformer((ohe, ['cat1', 'cat2']), (si, ['ord1', 'ord2']), remainder='passthrough')
ct.fit_transform(df_train)


#Define Validation method
#Train and validation set split
from sklearn.model_selection import train_test_split
X = df.drop('target_var', axis=1)
y = df['column to predict']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, stratify = y.values, random_state = 101)
#Cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)
#KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=101)
for train_index, val_index in kf.split(X, y):
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
#StratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=101)
for train_index, val_index in skf.split(X, y):
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
#Leave-One-Out Cross validation
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

#Select columns of a certain type
df_bool = df.select_dtypes(include='bool')
df_noint = df.select_dtypes(exclude='int')

#Check for missing data
total_null = df.isna().sum().sort_values(ascending=False)
percent = 100*(df.isna().sum()/df.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#Generate new features with missing data
nanf = ['feature1', 'feature2', 'feature3']
for feature in nanf:
	df[feature + '_nan'] = df[feature].isna()
#Also look for infinite data, recommended to check it also after feature engineering
df.replace(np.inf,0,inplace=True)
df.replace(-np.inf,0,inplace=True)

#Check for duplicated data
df.duplicated().value_counts()
df['duplicated'] = df.duplicated() #Create a new feature

#Fill missing data 
df.fillna()
#Also with sklearn
from sklearn.impute import SimpleImputer
si = SimpleImputer()
imputed_X_train = pd.DataFrame(si.fit_transform(X_train))
imputed_X_val = pd.DataFrame(si.transform(X_val))
#Find nan values and fill them however you want
for i in df.iterrows():
	if np.isnan(i[1][col]):
		df[col].loc[i[0]] = i*3
#Fill NaNs generating predictions with KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform(X_train)

#Drop columns/rows
df.drop('column_full_of_nans')
df.dropna(how='any', inplace=True)

#Fix Skewed features
from scipy.stats import skew
numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
#Box Cox Transformation of (highly) skewed features. We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x
#Note that setting λ=0 is equivalent to log1p
from scipy.special import boxcox1p
skewed_features = skewness.index
lambd = 0.15
for feat in skewed_features:
	df[feat] = boxcox1p(df[feat], lambd)
#check different approaches to fix skewness:
skewed_features = skewness.index
for feature in skewed_features:
	original_skewness = skewness.loc[feature]['Skew']
	try:
		log_transform = skew(df[feature].apply(log1p))
		lambd = 0.15
		boxcox_transform = skew(boxcox1p(df[feature], lambd))
		print(f'{feature}')
		print(f'Original skewness: {original_skewness}')
		print(f'log1p transform skewness: {log_transform}')
		print(f'boxcox1p transform: {boxcox_transform}')
		print(f'Log1p Change: {original_skewness-log_transform}')
		print(f'BoxCox1p Change: {original_skewness-boxcox_transform}')
		if original_skewness > 0.5:
			if log_transform < 0.5 and log_transform > -0.5:
				print(f'Feature: {feature}, original skewness: {original_skewness}, new skewness: {log_transform}')
			if original_skewness-log_transform > 0.5:
				print(f'big change with feature {feature}, change: {original_skewness-log_transform}, new skewness: {log_transform}')
		if original_skewness < -0.5:
			if log_transform < 0.5 and log_transform > -0.5:
				print(f'Feature: {feature}, original skewness: {original_skewness}, new skewness: {log_transform}')
			if original_skewness-log_transform > 0.5:
				print(f'big change with feature {feature}, change: {original_skewness-log_transform}, new skewness: {log_transform}')

#Exploratory Data Analysis (EDA)
sns.pairplot(df, hue='categorical_var')
sns.distplot(df['column'])
sns.countplot(df['column'])

#Feature understanding - see how the variable affects the target variable
from featexp import get_univariate_plots
# Plots drawn for all features if nothing is passed in feature_list parameter.
get_univariate_plots(data=data_train, target_col='target', features_list=['DAYS_BIRTH'], bins=10)
get_univariate_plots(data=data_train, target_col='target', data_test=data_test, features_list=['DAYS_EMPLOYED'])
from featexp import get_trend_stats
stats = get_trend_stats(data=data_train, target_col='target', data_test=data_test)

#Fix or remove outliers
sns.boxplot(df['feature1'])
sns.boxplot(df['feature2'])
plt.scatter('var1', 'y') #Do this for all variables against y

def replace_outlier(df, column, value, threshold, direction='max'): #value could be the mean
		if direction == 'max':
			df[column] = df[column].apply(lambda x: value if x > threshold else x)
			for item in df[df[column] > threshold].index:
				df.loc[item, (column+'_nan')] = 1
		elif direction == 'min':
			df[column] = df[column].apply(lambda x: value if x < threshold else x)
			for item in df[df[column] < threshold].index:
				df.loc[item, (column+'_nan')] = 1

#Outlier detection with Isolation Forest
from sklearn.ensemble import IsolationForest
anomalies_ratio = 0.009
isolation_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=anomalies_ratio, behaviour='new', random_state=101)
isolation_forest.fit(df)
outliers = isolation_forest.predict(df)
outliers = [1 if x == -1 else 0 for x in outliers]
df['Outlier'] = outliers

#Outlier detection with Mahalanobis Distance
def is_pos_def(A):
	if np.allclose(A, A.T):
		try:
			np.linalg.cholesky(A)
			return True
		except np.linalg.LinAlgError:
			return False
	else:
		return False

def cov_matrix(data):
	covariance_matrix = np.cov(data, rowvar=False)
	if is_pos_def(covariance_matrix):
		inv_covariance_matrix = np.linalg.inv(covariance_matrix)
		if is_pos_def(inv_covariance_matrix):
			return covariance_matrix, inv_covariance_matrix
		else:
			print('Error: Inverse Covariance Matrix is not positive definite')
	else:
		print('Error: Covariance Matrix is not positive definite')

def mahalanobis_distance(inv_covariance_matrix, data):
		normalized = data - data.mean(axis=0)
		md = []
		for i in range(len(normalized)):
			md.append(np.sqrt(normalized[i].dot(inv_covariance_matrix).dot(normalized[i])))
		return md

#Mahalanobis Distance should follow X2 distribution, let's visualize it:
sns.distplot(np.square(dist), bins=10, kde=False)

def mahalanobis_distance_threshold(dist, k=2): #k=3 for a higher threshold
	return np.mean(dist)*k

#Visualize the Mahalanobis distance to check if the threshold is reasonable
sns.distplot(dist, bins=10, kde=True)

def mahalanobis_distance_detect_outliers(dist, k=2):
	threshold = mahalanobis_distance_threshold(dist, k)
	outliers = []
	for i in range(len(dist)):
		if dist[i] >= threshold:
			outliers.append(i) #index of the outlier
	return np.array(outliers)

md = mahalanobis_distance_detect_outliers(mahalanobis_distance(cov_matrix(df)[1], df), k=2)
#Flag outliers with Mahalanobis Distance
threshold = mahalanobis_distance_threshold(mahalanobis_distance(cov_matrix(df)[1], df), k=2)
outlier = pd.DataFrame({'Mahalanobis distance': md, 'Threshold':threshold})
outlier['Outlier'] = outlier[outlier['Mahalanobis distance'] > outlier['Threshold']]
df['Outlier'] = outlier['Outlier']

#Correlation analysis
sns.heatmap(df.corr(), annot=True, fmt='.2f')
correlations = df.corr(method='pearson').abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]

#Colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor    
def calculate_vif_(X, thresh=5.0):
	variables = list(range(X.shape[1]))
	dropped = True
	while dropped:
		dropped = False
		vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
			   for ix in range(X.iloc[:, variables].shape[1])]

		maxloc = vif.index(max(vif))
		if max(vif) > thresh:
			print('vif ' + vif + ' dropping \'' + X.iloc[:, variables].columns[maxloc] +
				  '\' at index: ' + str(maxloc))
			del variables[maxloc]
			dropped = True

	print('Remaining variables:')
	print(X.columns[variables])
	return X.iloc[:, variables]

#Encode categorical variables
#Encoding for target variable (categorical variable)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['categorical_var'] = le.fit_transform(df['categorical_var'])
#Check for new categories in the validation/test set
X_val[col] = X_val[col].apply(lambda x: 'new_category' if x not in le.classes_ else x)
le.classes_ = np.append(le.classes_, 'new_category')

#One hot encoding for categorical information
#Use sklearn's OneHotEncoder for categories encoded as possitive real numbers
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
df['var_to_encode'] = enc.fit_transform(df['var_to_encode'])
#Also can be done like this
OH_cols_train = pd.DataFrame(enc.fit_transform(X_train[low_cardinality_cols]))
OH_cols_val = pd.DataFrame(enc.transform(X_val[low_cardinality_cols]))
OH_cols_train.index = X_train.index
OH_cols_val.index = X_val.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(low_cardinality_cols, axis=1)
num_X_val = X_val.drop(low_cardinality_cols, axis=1)
# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_val = pd.concat([num_X_valid, OH_cols_val], axis=1)

#Use pandas get_dummies for categories encoded as strings
pd.get_dummies(df, columns=['col1','col2'])

#OrdinalEncoding for categories which have an order (example: low/medium/high)
map_dict = {'low': 0, 'medium': 1, 'high': 2}
df['var_oe'] = df['var'].apply(lambda x: map_dict[x])
#We can also do it with sklearn's LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['var_oe'] = le.fit_transform(df['var'])

#BinaryEncoder when we have many categories in one variable it means creating many columns with OHE. With Binary encoding we can do so with many less columns by using binary numbers. Use only when there is a high cardinality in the categorical variable.
from category_encoders.binary import BinaryEncoder
be = BinaryEncoder(cols = ['var'])
df = be.fit_transform(df)

#HashingEncoder
from category_encoders.hashing import HashingEncoder
he = HashingEncoder(cols = ['var'])
df = he.fit_transform(df)

#CountEncoding. Counts the number of times a category appears and assigns it as the value for the category. It works because rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping.
from category_encoders import CountEncoder
ce = CountEncoder()
df['col_CE'] = ce.fit_transform(df['col'])

#TargetEncoding. calculate the average outcome for all the rows with the same category and assigns it as the value for the category
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['col1', 'col2', 'col3'])
te.fit(df[['col1', 'col2', 'col3']], df['target'])

#CatBoostEncoding. This is similar to target encoding in that it's based on the target probablity for a given value. However with CatBoost, for each row, the target probability is calculated only from the rows before it.
from category_encoders import CatBoostEncoder
cbe = CatBoostEncoder(cols=['col1', 'col2', 'col3'])
cbe.fit(df[['col1', 'col2', 'col3']], df['target'])

#Resampling Methods for Unbalanced Datasets https://towardsdatascience.com/https-towardsdatascience-com-resampling-methods-for-unbalanced-datasets-5b565d0a247d
#Undersampling the majority class is taking random draws of the dominating class out of the dataset to match the amount of non-dominating class. As a general rule, this is usually the least desirable approach as it causes us to lose some valuable data by throwing it away, but when you have a large dataset, it might prove to be computationally better to undersample.
#Oversampling the minority class is the opposite. Instead of the previous approach, we take random draws of the non-dominating class and create “fake” copies to match the amount of cases in the dominating class. In this case, we are in essence creating duplicates of the data and training our model on such. This may not be an ideal approach when our non-dominating class is not scattered across the dataset. Duplication will effectively only recreate similar instances without a “synthetic” variety.
#Synthetic Minority Oversampling Technique (SMOTE) is another type of minority oversampling technique, except that this one takes into account characteristics of existing cases of non dominating class, and creates synthetic duplicates in a “nearest neighbors” fashion
from imblearn.over_sampling import SMOTE
method = SMOTE(kind='regular')
X_resampled, y_resampled = method.fit_sample(X_train, y_train)
#Undersampling with Tomek Links. Tomek links are pairs of examples of opposite classes in close vicinity. https://towardsdatascience.com/the-5-most-useful-techniques-to-handle-imbalanced-datasets-6cdba096d55a
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

#Feature selection: Drop attributes that provide no useful information for the task
#Unsupervised Feature selection before training a model
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k='all')
selected = bestfeatures.fit(X_train, y_train)

dfscores = pd.DataFrame(selected.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

featureScores.sort_values('Score', ascending=False) #The highest the number, the more irrelevant the variable is
#Select Percentile
from sklearn.feature_selection import SelectPercentile
bestfeatures = SelectPercentile(percentile=50)
selected = bestfeatures.fit(X_train, y_train)
X_train_selected = selected.transform(X_train)
#Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
X_train_selected = selector.fit_transform(X_train)

#Feature engineering. Create new features by transforming the data
#Discretize continuous features
#Decompose features (categorical, date/time, etc.)
df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
df['second'] = df['timestamp'].dt.second.astype('uint8')
df['minute'] = df['timestamp'].dt.minute.astype('uint8')
df['hour'] = df['timestamp'].dt.hour.astype('uint8')
df['day'] = df['timestamp'].dt.day.astype('uint8')
df['dayofweek'] = df['timestamp'].dt.dayofweek.astype('uint8')
df['dayofyear'] = df['timestamp'].dt.dayofyear.astype('uint8')
df['weekofyear'] = df['timestamp'].dt.weekofyear.astype('uint8')
df['month'] = df['timestamp'].dt.month.astype('uint8')
df['year'] = df['timestamp'].dt.year.astype('uint8')
#Create periodic features from date with sine and cosine
cycle = 7
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / cycle)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / cycle)
#Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.)
#Aggregate features into promising new features (x*y)
#Combine categorical variables
df['col1_col2'] = df['col1'] + '_' + df['col2']
#For speed/movement data, add vectorial features. Try many different combinations
df['position_norm'] = df['position_X'] ** 2 + df['position_Y'] ** 2 + df['position_Z'] ** 2
df['position_module'] = df['position_norm'] ** 0.5
df['position_norm_X'] = df['position_X'] / df['position_module']
df['position_norm_Y'] = df['position_Y'] / df['position_module']
df['position_norm_Z'] = df['position_Z'] / df['position_module']
df['position_over_velocity'] = df['position_module'] / df['velocity_module']
#For time series data: Discretize the data by different samples.
from astropy.stats import median_absolute_deviation
from statsmodels.robust.scale import mad
from scipy.stats import kurtosis
from scipy.stats import skew

def CPT5(x):
	den = len(x)*np.exp(np.std(x))
	return sum(np.exp(x))/den

def SSC(x):
	x = np.array(x)
	x = np.append(x[-1], x)
	x = np.append(x, x[1])
	xn = x[1:len(x)-1]
	xn_i2 = x[2:len(x)]    #xn+1
	xn_i1 = x[0:len(x)-2]  #xn-1
	ans = np.heaviside((xn-xn_i1)*(xn-xn_i2), 0)
	return sum(ans[1:])

def wave_length(x):
	x = np.array(x)
	x = np.append(x[-1], x)
	x = np.append(x, x[1])
	xn = x[1:len(x)-1]
	xn_i2 = x[2:len(x)]    #xn+1
	return sum(abs(xn_i2-xn))

def norm_entropy(x):
	tresh = 3
	return sum(np.power(abs(x), tresh))

def SRAV(x):
	SRA = sum(np.sqrt(abs(x)))
	return np.power(SRA/len(x), 2)

def mean_abs(x):
	return sum(abs(x))/len(x)

def zero_crossing(x):
	x = np.array(x)
	x = np.append(x[-1], x)
	x = np.append(x, x[1])
	xn = x[1:len(x)-1]
	xn_i2 = x[2:len(x)]    #xn+1
	return sum(np.heaviside(-xn*xn_i2, 0))

df_tmp = pd.DataFrame()
for column in tqdm(df.columns):
	df_tmp[column + '_mean'] = df.groupby(['series_id'])[column].mean()
	df_tmp[column + '_median'] = df.groupby(['series_id'])[column].median()
	df_tmp[column + '_max'] = df.groupby(['series_id'])[column].max()
	df_tmp[column + '_min'] = df.groupby(['series_id'])[column].min()
	df_tmp[column + '_std'] = df.groupby(['series_id'])[column].std()
	df_tmp[column + '_range'] = df_tmp[column + '_max'] - df_tmp[column + '_min']
	df_tmp[column + '_max_over_Min'] = df_tmp[column + '_max'] / df_tmp[column + '_min']
	df_tmp[column + 'median_abs_dev'] = df.groupby(['series_id'])[column].mad()
	df_tmp[column + '_mean_abs_chg'] = df.groupby(['series_id'])[column].apply(lambda x: np.mean(np.abs(np.diff(x))))
	df_tmp[column + '_mean_change_of_abs_change'] = df.groupby('series_id')[column].apply(lambda x: np.mean(np.diff(np.abs(np.diff(x)))))
	df_tmp[column + '_abs_max'] = df.groupby(['series_id'])[column].apply(lambda x: np.max(np.abs(x)))
	df_tmp[column + '_abs_min'] = df.groupby(['series_id'])[column].apply(lambda x: np.min(np.abs(x)))
	df_tmp[column + '_abs_avg'] = (df_tmp[column + '_abs_min'] + df_tmp[column + '_abs_max'])/2
	df_tmp[column + '_abs_mean'] = df.groupby('series_id')[column].apply(lambda x: np.mean(np.abs(x)))
	df_tmp[column + '_abs_std'] = df.groupby('series_id')[column].apply(lambda x: np.std(np.abs(x)))
	df_tmp[column + '_abs_range'] = df_tmp[column + '_abs_max'] - df_tmp[column + '_abs_min']
	df_tmp[column + '_skew'] = df.groupby(['series_id'])[column].skew()
	df_tmp[column + '_q25'] = df.groupby(['series_id'])[column].quantile(0.25)
	df_tmp[column + '_q75'] = df.groupby(['series_id'])[column].quantile(0.75)
	df_tmp[column + '_q95'] = df.groupby(['series_id'])[column].quantile(0.95)
	df_tmp[column + '_iqr'] = df_tmp[column + '_q75'] - df_tmp[column + '_q25']
	df_tmp[column + '_CPT5'] = df.groupby(['series_id'])[column].apply(CPT5)
	df_tmp[column + '_SSC'] = df.groupby(['series_id'])[column].apply(SSC)
	df_tmp[column + '_wave_lenght'] = df.groupby(['series_id'])[column].apply(wave_length)
	df_tmp[column + '_norm_entropy'] = df.groupby(['series_id'])[column].apply(norm_entropy)
	df_tmp[column + '_SRAV'] = df.groupby(['series_id'])[column].apply(SRAV)
	df_tmp[column + '_kurtosis'] = df.groupby(['series_id'])[column].apply(kurtosis)
	df_tmp[column + '_zero_crossing'] = df.groupby(['series_id'])[column].apply(zero_crossing)
	df_tmp[column +  '_unq'] = df[column].round(3).nunique()
	try:
		df_tmp[column + '_freq'] = df[column].value_counts().idxmax()
	except:
		df_tmp[column + '_freq'] = 0
	df_tmp[column + '_max_freq'] = df[df[column] == df[column].max()].shape[0]
	df_tmp[column + '_min_freq'] = df[df[column] == df[column].min()].shape[0]
	df_tmp[column + '_pos_freq'] = df[df[column] >= 0].shape[0]
	df_tmp[column + '_neg_freq'] = df[df[column] < 0].shape[0]
	df_tmp[column + '_nzeros'] = (df[column]==0).sum(axis=0)
df = df_tmp.copy()
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
#Create new column, dividing the dataset in groups of the same number of events
df['group'] = pd.cut(df['Age'], 3, labels=['kids', 'adults', 'senior'])
#Filter dataframe. Only values in a list
df[df['column'].isin(values_list)]
#Tabular data Feature engineering with Deltapy
from deltapy import transform, interact, mapper, extract
#Transformation
df_out = transform.robust_scaler(df.copy(), drop=["Close_1"])
df_out = transform.standard_scaler(df.copy(), drop=["Close"])
df_out = transform.fast_fracdiff(df.copy(), ["Close","Open"],0.5)
df_out = transform.windsorization(df.copy(),"Close",para,strategy='both')
df_out = transform.operations(df.copy(),["Close"])
df_out = transform.triple_exponential_smoothing(df.copy(),["Close"], 12, .2,.2,.2,0)
df_out = transform.naive_dec(df.copy(), ["Close","Open"])
df_out = transform.bkb(df.copy(), ["Close"])
df_out = transform.butter_lowpass_filter(df.copy(),["Close"],4)
df_out = transform.instantaneous_phases(df.copy(), ["Close"])
df_out = transform.kalman_feat(df.copy(), ["Close"])
df_out = transform.perd_feat(df.copy(),["Close"])
df_out = transform.fft_feat(df.copy(), ["Close"])
df_out = transform.harmonicradar_cw(df.copy(), ["Close"],0.3,0.2)
df_out = transform.saw(df.copy(),["Close","Open"])
df_out = transform.modify(df.copy(),["Close"])
df_out = transform.multiple_rolling(df, columns=["Close"])
df_out = transform.multiple_lags(df, start=1, end=3, columns=["Close"])
df_out  = transform.prophet_feat(df.copy().reset_index(),["Close","Open"],"Date", "D")
#Interaction
df_out = interact.lowess(df.copy(), ["Open","Volume"], df["Close"], f=0.25, iter=3)
df_out = interact.autoregression(df.copy())
df_out = interact.muldiv(df.copy(), ["Close","Open"])
df_out = interact.decision_tree_disc(df.copy(), ["Close"])
df_out = interact.quantile_normalize(df.copy(), drop=["Close"])
df_out = interact.tech(df.copy())
df_out = interact.genetic_feat(df.copy())
#Mapping
df_out = mapper.pca_feature(df.copy(),variance_or_components=0.80,drop_cols=["Close_1"])
df_out = mapper.cross_lag(df.copy())
df_out = mapper.a_chi(df.copy())
df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15)
df_out = mapper.lle_feat(df.copy(),["Close_1"],4)
df_out = mapper.feature_agg(df.copy(),["Close_1"],4 )
df_out = mapper.neigh_feat(df.copy(),["Close_1"],4 )
#Extraction
extract.abs_energy(df["Close"])
extract.cid_ce(df["Close"], True)
extract.mean_abs_change(df["Close"])
extract.mean_second_derivative_central(df["Close"])
extract.variance_larger_than_standard_deviation(df["Close"])
extract.var_index(df["Close"].values,var_index_param)
extract.symmetry_looking(df["Close"])
extract.has_duplicate_max(df["Close"])
extract.partial_autocorrelation(df["Close"])
extract.augmented_dickey_fuller(df["Close"])
extract.gskew(df["Close"])
extract.stetson_mean(df["Close"])
extract.length(df["Close"])
extract.count_above_mean(df["Close"])
extract.longest_strike_below_mean(df["Close"])
extract.wozniak(df["Close"])
extract.last_location_of_maximum(df["Close"])
extract.fft_coefficient(df["Close"])
extract.ar_coefficient(df["Close"])
extract.index_mass_quantile(df["Close"])
extract.number_cwt_peaks(df["Close"])
extract.spkt_welch_density(df["Close"])
extract.linear_trend_timewise(df["Close"])
extract.c3(df["Close"])
extract.binned_entropy(df["Close"])
extract.svd_entropy(df["Close"].values)
extract.hjorth_complexity(df["Close"])
extract.max_langevin_fixed_point(df["Close"])
extract.percent_amplitude(df["Close"])
extract.cad_prob(df["Close"])
extract.zero_crossing_derivative(df["Close"])
extract.detrended_fluctuation_analysis(df["Close"])
extract.fisher_information(df["Close"])
extract.higuchi_fractal_dimension(df["Close"])
extract.petrosian_fractal_dimension(df["Close"])
extract.hurst_exponent(df["Close"])
extract.largest_lyauponov_exponent(df["Close"])
extract.whelch_method(df["Close"])
extract.find_freq(df["Close"])
extract.flux_perc(df["Close"])
extract.range_cum_s(df["Close"])
extract.structure_func(df["Close"])
extract.kurtosis(df["Close"])
extract.stetson_k(df["Close"])

#Scaling features
#Standard Scaler: The StandardScaler assumes your data is normally distributed within each feature and will scale them such that the distribution is now centred around 0, with a standard deviation of 1.
#If data is not normally distributed, this is not the best scaler to use.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
#MinMax Scaler: Shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).
#This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.
#it is sensitive to outliers, so if there are outliers in the data, you might want to consider the Robust Scaler below.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
#Robust Scaler: Uses a similar method to the Min-Max scaler but it instead uses the interquartile range, rather than the min-max, so that it is robust to outliers.
#This means it is using less data for scaling so it’s more suitable when there are outliers in the data.RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
#Normalizer: The normalizer scales each value by dividing it by its magnitude in n-dimensional space for n number of features.
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
#PowerTransformer: make data more Gaussian-like
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_train = pt.fit_transform(X_train)
#QuantileTransformer: spread the most frequent values and reduce impact of outliers
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer()
X_train = qt.fit_transform(X_train)

#Apply all the same transformations to the test set

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
# Lasso Regression
#########
#Reduces the sum of the absolute values of the coefficients, so every variable has less influence. Some coefficients will be zero, removing some variables from the model. Reduces overfitting compared to linear regression
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.0005, random_state=101)
ls.fit(X_train,y_train)

y_pred = ls.predict(X_val)

#########
# Ridge Regression
#########
#Makes coefficients smaller, so every variable has less influence. Reduces overfitting compared to linear regression
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=0.002, random_state=101)
rdg.fit(X_train,y_train)

y_pred = rdg.predict(X_val)

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
# Linear SVC
#########
from sklearn.svm import LinearSVC
lSVC = LinearSVC()
lSVC.fit(X_train, y_train)

y_pred = lSVC.predict(X_val)

#########
# KNN
#########
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
#For regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
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
rfc = RandomForestClassifier(n_estimators=200, random_state=101, n_jobs=-1, verbose=3)
rfc.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=200, random_state=101, n_jobs=-1, verbose=3)
rfr.fit(X_train, y_train)

#Use model to predict
y_pred = rfr.predict(X_val)

#Evaluate accuracy of the model
acc_rf = round(rfr.score(X_val, y_val) * 100, 2)

#Evaluate feature importance
importances = rfr.feature_importances_
std = np.std([importances for tree in rfr.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame(rfr.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances.sort_values('importance', ascending=False)

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#########
# lightGBM (LGBM)
#########
import lightgbm as lgb
#create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

#specify your configurations as a dict
params = {
	'boosting_type': 'gbdt',
	'objective': 'regression',
	'metric': {'l2', 'l1'},
	'num_leaves': 31,
	'learning_rate': 0.05,
	'feature_fraction': 0.9,
	'bagging_fraction': 0.8,
	'bagging_freq': 5,
	'verbose': 0
}

#train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)

#save model to file
gbm.save_model('model.txt')

#predict
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

#########
# XGBoost
#########
import xgboost as xgb

params = {'objective': 'multi:softmax',  #Specify multiclass classification
		'num_class': 9,  #Number of possible output classes
		'tree_method': 'hist',  #Use gpu_hist for GPU accelerated algorithm.
		'eta': 0.1,
		'max_depth': 6,
		'silent': 1,
		'gamma': 0,
		'eval_metric': "merror",
		'min_child_weight': 3,
		'max_delta_step': 1,
		'subsample': 0.9,
		'colsample_bytree': 0.4,
		'colsample_bylevel': 0.6,
		'colsample_bynode': 0.5,
		'lambda': 0,
		'alpha': 0,
		'seed': 0}

xgtrain = xgb.DMatrix(X_train, label=y_train)
xgval = xgb.DMatrix(X_val, label=y_val)
xgtest = xgb.DMatrix(X_test)

num_rounds = 500
gpu_res = {}  #Store accuracy result
#Train model
xgbst = xgb.train(params, xgtrain, num_rounds, evals=[
			(xgval, 'test')], evals_result=gpu_res)

y_pred = xgbst.predict(xgtest)

#Simplified code
model = xgb.XGBClassifier(random_state=1, n_estimators=1000, learning_rate=0.01, n_jobs=-1) #for the best model, high number of estimators, low learning rate
model.fit(X_train, y_train)
model.score(X_val,y_val)
#Regression
model=xgb.XGBRegressor(random_state=1, n_estimators=1000, learning_rate=0.01, n_jobs=-1) #for the best model, high number of estimators, low learning rate
model.fit(X_train, y_train)
model.score(X_val,y_val)

#########
# AdaBoost
#########
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=101)
model.fit(X_train, y_train)
model.score(X_val,y_val)
#Regression
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(random_state=101)
model.fit(X_train, y_train)
model.score(X_val, y_val)

#########
# CatBoost
#########
#CatBoost algorithm works great for data with lots of categorical variables. You should not perform one-hot encoding for categorical variables before applying this model.
from catboost import CatBoostClassifier
model = CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_val, y_val))
model.score(X_val, y_val)
#Regression
from catboost import CatBoostRegressor
model = CatBoostRegressor()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_val, y_val))
model.score(X_val, y_val)

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
# Multi-layer Perceptron Regressor (Neural Network)
#########
from sklearn.neural_network import MLPRegressor

lr = 0.01 #Learning rate
nn = [2, 16, 8, 1] #Neurons by layer

MLPr = MLPRegressor(solver='sgd', learning_rate_init=lr, hidden_layer_sizes=tuple(nn[1:]), verbose=True, n_iter_no_change=1000, batch_size = 64)
MLPr.fit(X_train, y_train)
MLPr.predict(X_val)

#########
# Multi-layer Perceptron Classifier (Neural Network)
#########
from sklearn.neural_network import MLPClassifier

lr = 0.01 #Learning rate
nn = [2, 16, 8, 1] #Neurons by layer

MLPc = MLPClassifier(solver='sgd', learning_rate_init=lr, hidden_layer_sizes=tuple(nn[1:]), verbose=True, n_iter_no_change=1000, batch_size = 64)
MLPc.fit(X_train, y_train)
MLPc.predict(X_val)

#########
# AutoML
#########
import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML
h2o.init()

train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')

aml = H2OAutoML(max_models=50, max_runtime_secs=30000)
aml.train(x=train.columns, y='target', training_frame=train)

lb = aml.leaderboard
lb.head(rows=lb.nrows)

y_pred = aml.predict(test)

#########
# K-Means Clustering
#########
#Find parameter k: Elbow method
SSE = []
for k in range(1,10):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(df)
	SSE.append(kmeans.inertia_)

plt.plot(list(range(1,10)), SSE)
#Find parameter k: Silhouette score
from sklearn.metrics import silhouette_score
ss=[]
for k in range(1,10):
	kmeans = KMeans(n_clusters=k)
	ss.append(silhouette_score(df, kmeans.labels_))
plt.plot(list(range(1,10)), ss) #select k where silhouette score is max
#Train model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K) #Choose K
kmeans.fit(df)

#Evaluate the model
kmeans.cluster_centers_
kmeans.labels_

#########
# DBSCAN
#########
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5)
clusters = dbscan.fit_predict(X)
dbscan.labels_ #samples with label -1 are considered anomalies

#Measure and compare their performance
#Big thank you to Uxue Lazcano (https://github.com/uxuelazkano) for code on model comparison
models = pd.DataFrame({
'Model': ['Linear Regression', 'Support Vector Machine', 'KNN', 'Logistic Regression', 
			'Random Forest'],
'Score': [acc_lr, acc_svm, acc_knn, acc_log, 
			acc_rf]})
models.sort_values(by='Score', ascending=False)

#Evaluate how each model is working
plt.scatter(y_val, y_pred) #should have the shape of a line for good predictions
sns.distplot(y_val - y_pred) #should be a normal distribution centered at 0

#Save model
import pickle
pickle.dump(model, open("model.pkl", "wb"))
#Load model
model = pickle.load(open("model.pkl", "rb"))

#Analyze the most significant variables for each algorithm.
#Analyze the types of errors the models make.
#What data would a human have used to avoid these errors?
#Have a quick round of feature selection and engineering.
#Have one or two more quick iterations of the five previous steps.
#Short-list the top three to five most promising models, preferring models that make different types of errors.
#Define Performance Metrics
#ROC AUC for classification tasks
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
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
#MAPE, note this metric is not implemented in sklearn, it can be problematic because it can cause division by zero errors
np.mean(np.abs((y_true-y_pred)/y_true))*100
#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_val,y_pred))

#Fine-tune the hyperparameters using cross-validation
#Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?)
#Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(model, param_grid, verbose = 3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
#my implementation
trl = []
scl = []
for trees in range(100, 5000, 50):
	rfr = RandomForestRegressor(n_estimators=trees, random_state=101, n_jobs=-1, verbose=3, criterion='mse')
	rfr.fit(X_train, y_train)
	sc = eval_model(rfr, X_val, y_val)
	trl.append(trees)
	scl.append(sc)

pd.DataFrame({'trees': trl, 'score':scl}).sort_values('score')
plt.plot(trl, scl)

#Confidence intervals
from scipy import stats
confidence = 0.95
squared_errors = np.square(y_pred - y_val)
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_erros.mean(), scale=stats.sem(squared_errors)))
#other methods
from sklearn.utils import resample
n_bootstraps = 1000
bootstrap_X = []
bootstrap_y = []
for _ in range(n_bootstraps):
	sample_X, sample_y = resample(scaled_df, target)
	bootstrap_X.append(sample_X)
	bootstrap_y.append(sample_y)
from sklearn.linear_model import SGDRegressor
linear_regression_model = SGDRegressor(tol=.0001, eta0=.01)
coeffs = []
for i, data in enumerate(bootstrap_X):
	linear_regression_model.fit(data, bootstrap_y[i])
	coeffs.append(linear_regression_model.coef_)
#Analyze coeffs to view 
coeffs = pd.DataFrame(coeffs)
coeffs.describe()
plt.boxplot(coeffs)

#Determine if two distributions are significantly different using the Mann Whitney U Test. https://towardsdatascience.com/determine-if-two-distributions-are-significantly-different-using-the-mann-whitney-u-test-1f79aa249ffb
def mann_whitney_u_test(distribution_1, distribution_2):
	"""
	Perform the Mann-Whitney U Test, comparing two different distributions.
	Args:
	   distribution_1: List. 
	   distribution_2: List.
	Outputs:
		u_statistic: Float. U statisitic for the test.
		p_value: Float.
	"""
	u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
	return u_statistic, p_value
mann_whitney_u_test(list(df['col1']), list(df['col2']))
#As a general rule of thumb, when the p-value is below 0.05, the null hypothesis can be rejected. This means with statistical significance that the two distributions are different.

#Try Ensemble methods. Combining your best models will often perform better than running them individually
#Max Voting
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3 = LogisticRegression()

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)

final_pred = np.array([])
for i in range(len(X_test)):
	final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))

#We can also use VotingClassifier from sklearn
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)

#Averaging
finalpred=(pred1+pred2+pred3)/3

#Weighted Average
finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)

#Stacking
from sklearn.model_selection import StratifiedKFold
def Stacking(model, train, y, test, n_fold):
	folds = StratifiedKFold(n_splits=n_fold, random_state=101)
	test_pred = np.empty((test.shape[0], 1), float)
	train_pred = np.empty((0, 1), float)
	for train_indices, val_indices in folds.split(train,y.values):
		X_train, X_val = train.iloc[train_indices], train.iloc[val_indices]
		y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

		model.fit(X_train, y_train)
		train_pred = np.append(train_pred, model.predict(X_val))
		test_pred = np.append(test_pred, model.predict(test))
	return test_pred.reshape(-1,1), train_pred

model1 = DecisionTreeClassifier(random_state=101)
test_pred1, train_pred1 = Stacking(model1, X_train, y_train, X_test, 10)
train_pred1 = pd.DataFrame(train_pred1)
test_pred1 = pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
test_pred2, train_pred2 = Stacking(model2, X_train, y_train, X_test, 10)
train_pred2 = pd.DataFrame(train_pred2)
test_pred2 = pd.DataFrame(test_pred2)

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=101)
model.fit(df,y_train)
model.score(df_test, y_test)

#Blending
model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
val_pred1 = pd.DataFrame(model1.predict(X_val))
test_pred1 = pd.DataFrame(model1.predict(X_test))

model2 = KNeighborsClassifier()
model2.fit(X_train,y_train)
val_pred2 = pd.DataFrame(model2.predict(X_val))
test_pred2 = pd.DataFrame(model2.predict(X_test))

df_val = pd.concat([X_val, val_pred1,val_pred2],axis=1)
df_test = pd.concat([X_test, test_pred1,test_pred2],axis=1)
model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)

#Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
ens = BaggingClassifier(DecisionTreeClassifier(random_state=101))
ens.fit(X_train, y_train)
ens.score(X_val,y_val)
#Regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
ens = BaggingRegressor(DecisionTreeRegressor(random_state=101))
ens.fit(X_train, y_train)
ens.score(X_val,y_val)

#Once you are confident about your final model, measure its performance on the test set to estimate the generalization error

#Model interpretability
#Feature importance
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=101).fit(X_val, y_val)
eli5.show_weights(perm, feature_names = X_val.columns.tolist())

#Partial dependence plot
#New integration in sklearn, might not work with older versions
from sklearn.inspection import partial_dependence, plot_partial_dependence
partial_dependence(model, X_train, features=['feature', ('feat1', 'feat2')])
plot_partial_dependence(model, X_train, features=['feature', ('feat1', 'feat2')])
#With external module for legacy editions
from pdpbox import pdp, get_dataset, info_plots

#Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_val, model_features=X_val.columns, feature='Goals Scored')

#plot it
pdp.pdp_plot(pdp_goals, 'Goals Scored')
plt.show()

#Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goals Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=model, dataset=X_val, model_features=X_val.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

#ALE Plots: faster and unbiased alternative to partial dependence plots (PDPs). They have a serious problem when the features are correlated.
#The computation of a partial dependence plot for a feature that is strongly correlated with other features involves averaging predictions of artificial data instances that are unlikely in reality. This can greatly bias the estimated feature effect.
#https://github.com/blent-ai/ALEPython

#SHAP Values: Understand how each feature affects every individual prediciton
import shap
data_for_prediction = X_val.iloc[row_num]
explainer = shap.TreeExplainer(model)  #Use DeepExplainer for Deep Learning models, KernelExplainer for all other models
shap_vals = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_vals[1], data_for_prediction)

#We can also do a SHAP plot of the whole dataset
shap_vals = explainer.shap_values(X_val)
shap.summary_plot(shap_vals[1], X_val)
#SHAP Dependence plot
shap.dependence_plot('feature_for_x', shap_vals[1], X_val, interaction_index="feature_for_color")

#Local interpretable model-agnostic explanations (LIME)
#Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions.
#https://github.com/marcotcr/lime 

#Dimensionality reduction
#SVD: Find the percentage of variance explained by each principal component
#First scale the data
U, S, V = np.linalg.svd(df, full_matrices=False)
importance = S/S.sum()
varinace_explained = importance.cumsum()*100
#PCA: Decompose the data in a defined number of variables keeping the most variance possible.
#Important to standarize the data, we're looking to find the variables that maximize the variance within the data.
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full') #Choose number of components so that 99% of the variance is retained
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.index = X_train.index

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(X_test_PCA)
X_test_PCA.index = X_test.index
#NMF
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=101)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_val_nmf = nmf.transform(X_val)
#t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(random_state=101)
X_tsne = tsne.fit_transform(X)

#ONLY FOR KAGGLE, NOT FOR REAL LIFE PROBLEMS
#If both train and test data come from the same distribution use this, we can use the target variable averaged over different categorical variables as a feature.
from sklearn import base
from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
	def __init__(self,colnames,targetName, n_fold=5, verbosity=True, discardOriginal_col=False):
		self.colnames = colnames
		self.targetName = targetName
		self.n_fold = n_fold
		self.verbosity = verbosity
		self.discardOriginal_col = discardOriginal_col

	def fit(self, X, y=None):
		return self

	def transform(self,X):
		assert(type(self.targetName) == str)
		assert(type(self.colnames) == str)
		assert(self.colnames in X.columns)
		assert(self.targetName in X.columns)

		mean_of_target = X[self.targetName].mean()
		kf = KFold(n_splits = self.n_fold, shuffle = True, random_state=2019)

		col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
		X[col_mean_name] = np.nan

		for tr_ind, val_ind in kf.split(X):
			X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
			X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
			X[col_mean_name].fillna(mean_of_target, inplace = True)

		if self.verbosity:
			encoded_feature = X[col_mean_name].values
			print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,
			np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))

		if self.discardOriginal_col:
			X = X.drop(self.targetName, axis=1)

		return X

targetc = KFoldTargetEncoderTrain('column','target',n_fold=5)
new_df = targetc.fit_transform(df)

new_df[['column_Kfold_Target_Enc','column']]


#########
# Start of Deep Learning part
#########
# Default configuration for a DNN according to Hands-on Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron 2nd edition
# Kernel Initialization: He Initialization
# Activation function: ELU
# Normalization: None if shallow; Batch Normalization if deep
# Regularization: Early stopping (+l2 if needed)
# Optimizer: Momentum Optimization (or RMSProp or Nadam)
# Learning rate scheduler: 1cycle
# 
# If the network is a simple stack of dense layers, then it can self-normalize, and you should use this configuration:
# Kernel Initialization: LeCun Initialization
# Activation function: SELU
# Normalization: None (self-normalization)
# Regularization: Alpha dropout if needed
# Optimizer: Momentum Optimization (or RMSProp or Nadam)
# Learning rate scheduler: 1cycle
# Don't forget to normalize the input features
#
# If you need a sparse model, you can use l1 regularization (and optionally zero out the tiny weights after training). If you need an even sparser model, you can use the TensorFlow Model Optimization Toolkit. This will break self-normalization, so you should use the default configuration in this case.
# If you need a low-latency model, you may need to use fewer layers, fold the Batch Normalization layers into the previous layers, and possibly use a faster activation function such as leaky ReLU or just ReLU. Having a sparse model will also help. Finally, you may want to reduce the float precision from 32 bits to 16 or even 8 bits. Chcek out TensorFlow Model Optimization Toolkit.
# If you are building a risk-sensitive application, or inference latency is not very important in your application, you can use Monte Carlo Dropout to boost performance and get more reliable probability estimates, along with uncertainty estimates.


#########
# Deep Learning (Coursera Specialization)
#########
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
#Initialize parameters
def layer_sizes(X, Y):
	"""
	Arguments:
	X -- input dataset of shape (input size, number of examples)
	Y -- labels of shape (output size, number of examples)
	
	Returns:
	n_x -- the size of the input layer
	n_h -- the size of the hidden layer
	n_y -- the size of the output layer
	"""
	n_x = X.shape[0] # size of input layer
	n_h = 4
	n_y = Y.shape[1] # size of output layer
	return (n_x, n_h, n_y)
def initialize_parameters(layer_dims):
	"""
	Arguments:
	layer_dims -- python array (list) containing the dimensions of each layer in our network
	
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""
	parameters = {}
	L = len(layer_dims)            # number of layers in the network

	for l in range(1, L):
		#Random initialization, not the best
		#parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01

		#He initialization (recommended)
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
		
	return parameters
#Forward propagation
def linear_forward(A, W, b):
	"""
	Implement the linear part of a layer's forward propagation.

	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	Z = np.dot(W, A)+b
	
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python tuple containing "linear_cache" and "activation_cache";
			 stored for computing the backward pass efficiently
	"""
	
	if activation == "sigmoid":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	
	elif activation == "relu":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache
def forward_propagation(X, parameters):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	
	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
				every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""
	caches = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network
	
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
		caches.append(cache)
	
	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
	caches.append(cache)
	
	assert(AL.shape == (1,X.shape[1]))
			
	return AL, caches
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
	"""
	Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
	
	Arguments:
	X -- input dataset, of shape (2, number of examples)
	parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
					W1 -- weight matrix of shape (20, 2)
					b1 -- bias vector of shape (20, 1)
					W2 -- weight matrix of shape (3, 20)
					b2 -- bias vector of shape (3, 1)
					W3 -- weight matrix of shape (1, 3)
					b3 -- bias vector of shape (1, 1)
	keep_prob - probability of keeping a neuron active during drop-out, scalar
	
	Returns:
	A3 -- last activation value, output of the forward propagation, of shape (1,1)
	cache -- tuple, information stored for computing the backward propagation
	"""
	# retrieve parameters
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	
	# LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)
	D1 = np.random.randn(A1.shape[0], A1.shape[1])      # Step 1: initialize matrix D1 = np.random.rand(..., ...)
	D1 = D1 < keep_prob                                 # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
	A1 = A1*D1                                          # Step 3: shut down some neurons of A1
	A1 = A1/keep_prob                                   # Step 4: scale the value of neurons that haven't been shut down
	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)
	D2 = np.random.rand(A2.shape[0], A2.shape[1])       # Step 1: initialize matrix D2 = np.random.rand(..., ...)
	D2 = D2 < keep_prob                                 # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
	A2 = A2*D2                                          # Step 3: shut down some neurons of A2
	A2 = A2/keep_prob                                   # Step 4: scale the value of neurons that haven't been shut down
	Z3 = np.dot(W3, A2) + b3
	A3 = sigmoid(Z3)
	
	cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
	
	return A3, cache
#Cost function
def compute_cost(AL, Y):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m = Y.shape[1]

	# Compute loss from aL and y.
	cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
	
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost
def compute_cost_with_regularization(A3, Y, parameters, lambd):
	"""
	Implement the cost function with L2 regularization. See formula (2) above.
	
	Arguments:
	A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	parameters -- python dictionary containing parameters of the model
	
	Returns:
	cost - value of the regularized loss function (formula (2))
	"""
	m = Y.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	W3 = parameters["W3"]
	
	cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
	
	### START CODE HERE ### (approx. 1 line)
	L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
	### END CODER HERE ###
	
	cost = cross_entropy_cost + L2_regularization_cost
	
	return cost
#Backward propagation
def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, cache[0].T) / m
	db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
	dA_prev = np.dot(cache[1].T, dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (isinstance(db, float))
	
	return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
	dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db
def backward_propagation(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# Initializing the backpropagation
	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	current_cache = caches[-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
	
	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads
def backward_propagation_with_regularization(X, Y, cache, lambd):
	"""
	Implements the backward propagation of our baseline model to which we added an L2 regularization.
	
	Arguments:
	X -- input dataset, of shape (input size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	cache -- cache output from forward_propagation()
	lambd -- regularization hyperparameter, scalar
	
	Returns:
	gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
	"""
	
	m = X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
	dZ3 = A3 - Y
	
	dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3
	db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
	
	dA2 = np.dot(W3.T, dZ3)
	dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
	db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
	
	dA1 = np.dot(W2.T, dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
	db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
	
	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
				 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
				 "dZ1": dZ1, "dW1": dW1, "db1": db1}
	
	return gradients
# GRADED FUNCTION: backward_propagation_with_dropout

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
	"""
	Implements the backward propagation of our baseline model to which we added dropout.
	
	Arguments:
	X -- input dataset, of shape (2, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	cache -- cache output from forward_propagation_with_dropout()
	keep_prob - probability of keeping a neuron active during drop-out, scalar
	
	Returns:
	gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
	"""
	m = X.shape[1]
	(Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
	
	dZ3 = A3 - Y
	dW3 = 1./m * np.dot(dZ3, A2.T)
	db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
	dA2 = np.dot(W3.T, dZ3)
	dA2 = dA2*D2            # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
	dA2 = dA2/keep_prob     # Step 2: Scale the value of neurons that haven't been shut down
	### END CODE HERE ###
	dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	dW2 = 1./m * np.dot(dZ2, A1.T)
	db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
	
	dA1 = np.dot(W2.T, dZ2)
	dA1 = dA1*D1                # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
	dA1 = dA1/keep_prob         # Step 2: Scale the value of neurons that haven't been shut down
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = 1./m * np.dot(dZ1, X.T)
	db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
	
	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
				 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
				 "dZ1": dZ1, "dW1": dW1, "db1": db1}
	
	return gradients
#Optimization
def update_parameters_gradient_descent(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	"""
	L = len(parameters) // 2 # number of layers in the neural network

	# Update rule for each parameter. Use a for loop.
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
	return parameters
def random_mini_batches(X, Y, mini_batch_size = 64):
	"""
	Creates a list of random minibatches from (X, Y)
	
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	mini_batch_size -- size of the mini-batches, integer
	
	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""
	m = X.shape[1]                  # number of training examples
	mini_batches = []
		
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches
def initialize_velocity(parameters):
	"""
	Initializes the velocity as a python dictionary with:
				- keys: "dW1", "db1", ..., "dWL", "dbL" 
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	
	Returns:
	v -- python dictionary containing the current velocity.
					v['dW' + str(l)] = velocity of dWl
					v['db' + str(l)] = velocity of dbl
	"""
	L = len(parameters) // 2 # number of layers in the neural networks
	v = {}
	
	# Initialize velocity
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
		
	return v
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	"""
	Update parameters using Momentum
	
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar
	learning_rate -- the learning rate, scalar
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
	v -- python dictionary containing your updated velocities
	"""
	L = len(parameters) // 2 # number of layers in the neural networks
	
	# Momentum update for each parameter
	for l in range(L):
		# compute velocities
		v["dW" + str(l+1)] = beta*v["dW" + str(l+1)]+(1-beta)*grads["dW" + str(l+1)]
		v["db" + str(l+1)] = beta*v["db" + str(l+1)]+(1-beta)*grads["db" + str(l+1)]
		# update parameters
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
		
	return parameters, v
def initialize_adam(parameters) :
	"""
	Initializes v and s as two python dictionaries with:
				- keys: "dW1", "db1", ..., "dWL", "dbL" 
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters["W" + str(l)] = Wl
					parameters["b" + str(l)] = bl
	
	Returns: 
	v -- python dictionary that will contain the exponentially weighted average of the gradient.
					v["dW" + str(l)] = ...
					v["db" + str(l)] = ...
	s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
					s["dW" + str(l)] = ...
					s["db" + str(l)] = ...

	"""
	L = len(parameters) // 2 # number of layers in the neural networks
	v = {}
	s = {}
	
	# Initialize v, s. Input: "parameters". Outputs: "v, s".
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
		s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
	
	return v, s
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
								beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
	"""
	Update parameters using Adam
	
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	learning_rate -- the learning rate, scalar.
	beta1 -- Exponential decay hyperparameter for the first moment estimates 
	beta2 -- Exponential decay hyperparameter for the second moment estimates 
	epsilon -- hyperparameter preventing division by zero in Adam updates

	Returns:
	parameters -- python dictionary containing your updated parameters 
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	"""
	L = len(parameters) // 2                 # number of layers in the neural networks
	v_corrected = {}                         # Initializing first moment estimate, python dictionary
	s_corrected = {}                         # Initializing second moment estimate, python dictionary
	
	# Perform Adam update on all parameters
	for l in range(L):
		# Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
		v["dW" + str(l+1)] = beta1*v["dW" + str(l + 1)]+(1-beta1)*grads['dW' + str(l + 1)]
		v["db" + str(l+1)] = beta1*v["db" + str(l + 1)]+(1-beta1)*grads['db' + str(l + 1)]

		# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		v_corrected["dW" + str(l+1)] = v["dW" + str(l + 1)]/(1-np.power(beta1, t))
		v_corrected["db" + str(l+1)] = v["db" + str(l + 1)]/(1-np.power(beta1, t))

		# Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
		s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)]+(1-beta2)*np.power(grads['dW' + str(l+1)], 2)
		s["db" + str(l+1)] = beta2*s["db" + str(l+1)]+(1-beta2)*np.power(grads['db' + str(l+1)], 2)

		# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-np.power(beta2, t))
		s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-np.power(beta2, t))

		# Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*v_corrected["dW" + str(l+1)]/np.sqrt(s_corrected["dW" + str(l+1)]+epsilon)
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*v_corrected["db" + str(l+1)]/np.sqrt(s_corrected["db" + str(l+1)]+epsilon)

	return parameters, v, s
#Model
def nn_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, lambd = 0, keep_prob = 1, optimizer='adam', mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
	"""
	Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
	
	Arguments:
	X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
	layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
	learning_rate -- learning rate of the gradient descent update rule
	num_iterations -- number of iterations of the optimization loop
	print_cost -- if True, it prints the cost every 100 steps
	lambd -- regularization hyperparameter, scalar
	keep_prob - probability of keeping a neuron active during drop-out, scalar.
	
	Returns:
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""
	grads = {}
	costs = []          # keep track of cost
	m = X.shape[1]      # number of examples
	
	# Parameters initialization. (≈ 1 line of code)
	parameters = initialize_parameters(layers_dims)

	# Initialize the optimizer
	if optimizer == "gd":
		pass # no initialization required for gradient descent
	elif optimizer == "momentum":
		v = initialize_velocity(parameters)
	elif optimizer == "adam":
		v, s = initialize_adam(parameters)
	
	# Loop (gradient descent)
	for i in range(0, num_iterations):
		minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		cost_total = 0
		
		for minibatch in minibatches:
			# Select a minibatch
			(minibatch_X, minibatch_Y) = minibatch

			# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
			if keep_prob == 1:
				AL, caches = forward_propagation(X, parameters)
			elif keep_prob < 1:
				AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)

			# Compute cost and add to the cost total
			if lambd == 0:
				cost_total += compute_cost(AL, Y)
			else:
				cost_total += compute_cost_with_regularization(AL, Y, parameters, lambd)

			# Backward propagation
			assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
												# but this assignment will only explore one at a time
			if lambd == 0 and keep_prob == 1:
				grads = backward_propagation(minibatch_X, minibatch_Y, caches)
			elif lambd != 0:
				grads = backward_propagation_with_regularization(minibatch_X, minibatch_Y, caches, lambd)
			elif keep_prob < 1:
				grads = backward_propagation_with_dropout(minibatch_X, minibatch_Y, caches, keep_prob)

			# Update parameters
			if optimizer == "gd":
				parameters = update_parameters_gradient_descent(parameters, grads, learning_rate)
			elif optimizer == "momentum":
				parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer == "adam":
				t = t + 1 # Adam counter
				parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
															   t, learning_rate, beta1, beta2,  epsilon)
		cost_avg = cost_total / m
				
		# Print the cost every 100 training example
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
			
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	
	return parameters
#Predictions
def predict(parameters, X):
	"""
	Using the learned parameters, predicts a class for each example in X
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	X -- input data of size (n_x, m)
	
	Returns
	predictions -- vector of predictions of our model (red: 0 / blue: 1)
	"""
	# Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
	A2, cache = forward_propagation(X, parameters)
	predictions = np.round(A2)    
	return predictions
#########
# Convolutional Neural Networks
#########
#Padding a matrix
def zero_pad(X, pad):
	"""
	Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
	as illustrated in Figure 1.
	
	Argument:
	X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
	pad -- integer, amount of padding around each image on vertical and horizontal dimensions
	
	Returns:
	X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	"""
	X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = 0)
	
	return X_pad
#Apply a convolution
def conv_single_step(a_slice_prev, W, b):
	"""
	Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
	of the previous layer.
	
	Arguments:
	a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
	b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
	
	Returns:
	Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
	"""
	s = np.multiply(a_slice_prev, W)
	Z = np.sum(s)
	Z = Z + float(b)

	return Z
#Forward propagation for a convolution
def conv_forward(A_prev, W, b, hparameters):
	"""
	Implements the forward propagation for a convolution function
	
	Arguments:
	A_prev -- output activations of the previous layer, 
		numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
	b -- Biases, numpy array of shape (1, 1, 1, n_C)
	hparameters -- python dictionary containing "stride" and "pad"
		
	Returns:
	Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward() function
	"""
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(f, f, n_C_prev, n_C) = W.shape
	stride = hparameters['stride']
	pad = hparameters['pad']
	
	n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
	n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
	
	Z = np.zeros((m, n_H, n_W, n_C))
	A_prev_pad = zero_pad(A_prev, pad)
	
	for i in range(m):
		a_prev_pad = A_prev_pad[i]
		for h in range(n_H):
			vert_start = h*stride
			vert_end = vert_start + f
			for w in range(n_W):
				horiz_start = w*stride
				horiz_end = horiz_start+f
				for c in range(n_C):
					a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

					weights = W[...,c]
					biases = b[...,c]
					Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
	
	# Making sure your output shape is correct
	assert(Z.shape == (m, n_H, n_W, n_C))
	
	# Save information in "cache" for the backprop
	cache = (A_prev, W, b, hparameters)
	
	return Z, cache
#Forward propagation for a pooling
def pool_forward(A_prev, hparameters, mode = "max"):
	"""
	Implements the forward pass of the pooling layer
	
	Arguments:
	A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	hparameters -- python dictionary containing "f" and "stride"
	mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
	
	Returns:
	A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
	"""
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	f = hparameters["f"]
	stride = hparameters["stride"]
	
	n_H = int(1 + (n_H_prev - f) / stride)
	n_W = int(1 + (n_W_prev - f) / stride)
	n_C = n_C_prev
	
	A = np.zeros((m, n_H, n_W, n_C))              
	
	for i in range(m):
		for h in range(n_H):
			vert_start = h*stride
			vert_end = vert_start+f
			for w in range(n_W):
				horiz_start = w*stride
				horiz_end = horiz_start+f
				for c in range(n_C):
					a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
					
					if mode == "max":
						A[i, h, w, c] = np.max(a_prev_slice)
					elif mode == "average":
						A[i, h, w, c] = np.mean(a_prev_slice)
	
	# Store the input and hparameters in "cache" for pool_backward()
	cache = (A_prev, hparameters)
	
	# Making sure your output shape is correct
	assert(A.shape == (m, n_H, n_W, n_C))
	
	return A, cache
#########
# Deep Learning with TensorFlow
#########
import tensorflow as tf
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39
loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
												 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
	session.run(init)                            # Initializes the variables
	print(session.run(loss))                     # Prints the loss

#Use placeholder, then give values
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

#Example of cost function
z = tf.placeholder(tf.float32, name='z')
y = tf.placeholder(tf.float32, name='y')
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
sess = tf.Session()
cost = sess.run(cost, feed_dict={z: logits, y: labels}) #logits and labels defined somewhere else

#One-hot encoding with TensorFlow
C = tf.constant(C, name='C')
one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
sess = tf.Session()
one_hot = sess.run(one_hot_matrix)

#Create matrix full of ones or zeros
ones = tf.ones(shape)
zeros = tf.zeros(shape)

#Build a Neural Network with  TensorFlow
def create_placeholders(n_x, n_y):
	"""
	Creates the placeholders for the tensorflow session.
	
	Arguments:
	n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
	n_y -- scalar, number of classes (from 0 to 5, so -> 6)
	
	Returns:
	X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
	Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
	
	Tips:
	- You will use None because it let's us be flexible on the number of examples you will for the placeholders.
	  In fact, the number of examples during test/train is different.
	"""
	X = tf.placeholder(tf.float32, [n_x, None], name="X")
	Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
	
	return X, Y
def initialize_parameters():
	"""
	Initializes parameters to build a neural network with tensorflow. The shapes are:
						W1 : [25, 12288]
						b1 : [25, 1]
						W2 : [12, 25]
						b2 : [12, 1]
						W3 : [6, 12]
						b3 : [6, 1]
	
	Returns:
	parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
	"""
	W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2,
				  "W3": W3,
				  "b3": b3}
	
	return parameters
def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
	
	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
				  the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""
	# Retrieve the parameters from the dictionary "parameters" 
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	
	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	
	return Z3
def compute_cost(Z3, Y):
	"""
	Computes the cost
	
	Arguments:
	Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
	Y -- "true" labels vector placeholder, same shape as Z3
	
	Returns:
	cost - Tensor of the cost function
	"""
	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	
	return cost
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
	"""
	Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
	
	Arguments:
	X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
	Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
	X_test -- training set, of shape (input size = 12288, number of training examples = 120)
	Y_test -- test set, of shape (output size = 6, number of test examples = 120)
	learning_rate -- learning rate of the optimization
	num_epochs -- number of epochs of the optimization loop
	minibatch_size -- size of a minibatch
	print_cost -- True to print the cost every 100 epochs
	
	Returns:
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""
	(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
	n_y = Y_train.shape[0]                            # n_y : output size
	costs = []                                        # To keep track of the cost
	
	# Create Placeholders of shape (n_x, n_y)
	X, Y = create_placeholders(n_x, n_y)

	# Initialize parameters
	parameters = initialize_parameters()
	
	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z3 = forward_propagation(X, parameters)
	
	# Cost function: Add cost function to tensorflow graph
	cost = compute_cost(Z3, Y)
	
	# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	# Initialize all the variables
	init = tf.global_variables_initializer()

	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:
		sess.run(init)
		
		# Do the training loop
		for epoch in range(num_epochs):
			epoch_cost = 0.                       # Defines a cost related to an epoch
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:
				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch
				# IMPORTANT: The line that runs the graph on a minibatch.
				# Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			# Print the cost every epoch
			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
				
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per fives)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# lets save the parameters in a variable
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the correct predictions
		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
		
		return parameters
#Forward propagation for CNN
def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	
	Note that for simplicity and grading purposes, we'll hard-code some values
	such as the stride and kernel (filter) sizes. 
	Normally, functions should take these values as function parameters.
	
	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "W2"
				  the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""
	W1 = parameters['W1']
	W2 = parameters['W2']

	# CONV2D: stride of 1, padding 'SAME'
	Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
	# RELU
	A1 = tf.nn.relu(Z1)
	# MAXPOOL: window 8x8, stride 8, padding 'SAME'
	P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
	# CONV2D: filters W2, stride 1, padding 'SAME'
	Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
	# RELU
	A2 = tf.nn.relu(Z2)
	# MAXPOOL: window 4x4, stride 4, padding 'SAME'
	P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
	# FLATTEN
	F = tf.contrib.layers.flatten(P2)
	# FULLY-CONNECTED without non-linear activation function (not not call softmax).
	# 6 neurons in output layer.
	Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

	return Z3
#Cost for CNN
def compute_cost(Z3, Y):
	"""
	Computes the cost
	
	Arguments:
	Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
	Y -- "true" labels vector placeholder, same shape as Z3
	
	Returns:
	cost - Tensor of the cost function
	"""
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
	
	return cost
#CNN Model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
		  num_epochs = 100, minibatch_size = 64, print_cost = True):
	"""
	Implements a three-layer ConvNet in Tensorflow:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	
	Arguments:
	X_train -- training set, of shape (None, 64, 64, 3)
	Y_train -- test set, of shape (None, n_y = 6)
	X_test -- training set, of shape (None, 64, 64, 3)
	Y_test -- test set, of shape (None, n_y = 6)
	learning_rate -- learning rate of the optimization
	num_epochs -- number of epochs of the optimization loop
	minibatch_size -- size of a minibatch
	print_cost -- True to print the cost every 100 epochs
	
	Returns:
	train_accuracy -- real number, accuracy on the train set (X_train)
	test_accuracy -- real number, testing accuracy on the test set (X_test)
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""
	(m, n_H0, n_W0, n_C0) = X_train.shape             
	n_y = Y_train.shape[1]                            
	costs = []
	
	# Create Placeholders of the correct shape
	X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
	# Initialize parameters
	parameters = initialize_parameters()
	# Forward propagation: Build the forward propagation in the tensorflow graph
	Z3 = forward_propagation(X, parameters)
	# Cost function: Add cost function to tensorflow graph
	cost = compute_cost(Z3, Y)
	# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	# Initialize all the variables globally
	init = tf.global_variables_initializer()
	 
	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:
		# Run the initialization
		sess.run(init)
		# Do the training loop
		for epoch in range(num_epochs):
			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
			for minibatch in minibatches:
				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				minibatch_cost += temp_cost / num_minibatches

			# Print the cost every epoch
			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)
		
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# Calculate the correct predictions
		predict_op = tf.argmax(Z3, 1)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
		
		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(accuracy)
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
				
		return train_accuracy, test_accuracy, parameters
#########
# Deep learning with Keras
#########
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
def model(input_shape):
	"""
	input_shape: The height, width and channels as a tuple.  
		Note that this does not include the 'batch' as a dimension.
		If you have a batch like 'X_train', 
		then you can provide the input_shape using
		X_train.shape[1:]
	"""
	# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	X_input = Input(input_shape)

	# Zero-Padding: pads the border of X_input with zeroes
	X = ZeroPadding2D((3, 3))(X_input)

	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)

	# MAXPOOL
	X = MaxPooling2D((2, 2), name='max_pool')(X)

	# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fc')(X)

	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs = X_input, outputs = X, name='nn')

	return model
nn = model(X_train.shape[1:])
nn.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=40, batch_size=50)
preds = nn.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
nn.summary()
plot_model(nn, to_file='nn.png')
#DNN for tabular data
def tabular_dnn(numeric_variables, categorical_variables, categorical_counts,
                feature_selection_dropout=0.2, categorical_dropout=0.1,
                first_dense = 256, second_dense = 256, dense_dropout = 0.2, 
                activation_type=gelu):
    
    numerical_inputs = Input(shape=(len(numeric_variables),))
    numerical_normalization = BatchNormalization()(numerical_inputs)
    numerical_feature_selection = Dropout(feature_selection_dropout)(numerical_normalization)

    categorical_inputs = []
    categorical_embeddings = []
    for category in  categorical_variables:
        categorical_inputs.append(Input(shape=[1], name=category))
        category_counts = categorical_counts[category]
        categorical_embeddings.append(
            Embedding(category_counts+1, 
                      int(np.log1p(category_counts)+1), 
                      name = category + "_embed")(categorical_inputs[-1]))

    categorical_logits = Concatenate(name = "categorical_conc")([Flatten()(SpatialDropout1D(categorical_dropout)(cat_emb)) 
                                                                 for cat_emb in categorical_embeddings])

    x = concatenate([numerical_feature_selection, categorical_logits])
    x = Dense(first_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)  
    x = Dense(second_dense, activation=activation_type)(x)
    x = Dropout(dense_dropout)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model([numerical_inputs] + categorical_inputs, output)
    
    return model
def compile_model(model, loss, metrics, optimizer):
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model

model = tabular_dnn(numeric_variables, categorical_variables,
                        categorical_levels, 
                        feature_selection_dropout=0.1,
                        categorical_dropout=0.1,
                        first_dense = 256,
                        second_dense = 256,
                        dense_dropout = 0.1,
                        activation_type=gelu)
    
model = compile_model(model, binary_crossentropy, [AUC(name='auc')], Adam(learning_rate=0.0001))
#ResNet
def identity_block(X, f, filters, stage, block):
	"""
	Implementation of the identity block as defined in Figure 4
	
	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	f -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network
	
	Returns:
	X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
	"""
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	F1, F2, F3 = filters
	X_shortcut = X
	
	# First component of main path
	X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path
	X = Conv2D(F3, (f, f), strides = (1, 1), name = conv_name_base + '2b', padding = 'same', kernel_initializer=glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path
	X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer=glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	
	return X
def convolutional_block(X, f, filters, stage, block, s = 2):
	"""
	Implementation of the convolutional block as defined in Figure 4
	
	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	f -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network
	s -- Integer, specifying the stride to be used
	
	Returns:
	X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
	"""
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	F1, F2, F3 = filters
	X_shortcut = X

	##### MAIN PATH #####
	# First component of main path 
	X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path
	X = Conv2D(F2, (f, f), strides = (1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path
	X = Conv2D(F3, (1, 1), strides = (1,1), padding='valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

	##### SHORTCUT PATH ####
	X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding='valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X)
	X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	
	return X
def ResNet50(input_shape = (64, 64, 3), classes = 6):
	"""
	Implementation of the popular ResNet50 the following architecture:
	CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
	-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

	Arguments:
	input_shape -- shape of the images of the dataset
	classes -- integer, number of classes

	Returns:
	model -- a Model() instance in Keras
	"""
	X_input = Input(input_shape)
	X = ZeroPadding2D((3, 3))(X_input)
	
	# Stage 1
	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Stage 2
	X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
	X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
	X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

	# Stage 3
	X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

	# Stage 4
	X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

	# Stage 5
	X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
	X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
	X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

	# AVGPOOL
	X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

	# output layer
	X = Flatten()(X)
	X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
	
	# Create model
	model = Model(inputs = X_input, outputs = X, name='ResNet50')

	return model
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 2, batch_size = 32)
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#########
# YOLO for Object detecion
#########
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	"""Filters YOLO boxes by thresholding on object and class confidence.
	
	Arguments:
	box_confidence -- tensor of shape (19, 19, 5, 1)
	boxes -- tensor of shape (19, 19, 5, 4)
	box_class_probs -- tensor of shape (19, 19, 5, 80)
	threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
	
	Returns:
	scores -- tensor of shape (None,), containing the class probability score for selected boxes
	boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
	classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
	
	Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
	For example, the actual output size of scores would be (10,) if there are 10 boxes.
	"""
	# Step 1: Compute box scores
	box_scores = np.multiply(box_confidence, box_class_probs)
	
	# Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
	box_classes = K.argmax(box_scores, axis=-1)
	box_class_scores = K.max(box_scores, axis=-1)
	
	# Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	# same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
	filtering_mask = K.greater_equal(box_class_scores, threshold)
	
	# Step 4: Apply the mask to box_class_scores, boxes and box_classes
	scores = tf.boolean_mask(box_class_scores, filtering_mask)
	boxes = tf.boolean_mask(boxes, filtering_mask)
	classes = tf.boolean_mask(box_classes, filtering_mask)
	
	return scores, boxes, classes
#Intersection Over Union
# GRADED FUNCTION: iou

def iou(box1, box2):
	"""Implement the intersection over union (IoU) between box1 and box2
	
	Arguments:
	box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
	box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
	"""
	# Assign variable names to coordinates for clarity
	(box1_x1, box1_y1, box1_x2, box1_y2) = box1
	(box2_x1, box2_y1, box2_x2, box2_y2) = box2
	
	# Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
	xi1 = max(box1_x1, box2_x1)
	yi1 = max(box1_y1, box2_y1)
	xi2 = min(box1_x2, box2_x2)
	yi2 = min(box1_y2, box2_y2)
	inter_width = xi2-xi1
	inter_height = yi2-yi1
	inter_area = inter_width*inter_height

	# Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
	box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
	union_area = box1_area + box2_area - inter_area
	
	# compute the IoU
	iou = inter_area/union_area
	
	return iou
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
	"""
	Applies Non-max suppression (NMS) to set of boxes
	
	Arguments:
	scores -- tensor of shape (None,), output of yolo_filter_boxes()
	boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
	classes -- tensor of shape (None,), output of yolo_filter_boxes()
	max_boxes -- integer, maximum number of predicted boxes you'd like
	iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
	
	Returns:
	scores -- tensor of shape (, None), predicted score for each box
	boxes -- tensor of shape (4, None), predicted box coordinates
	classes -- tensor of shape (, None), predicted class for each box
	
	Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
	function will transpose the shapes of scores, boxes, classes. This is made for convenience.
	"""
	
	max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
	K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor

	nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
	
	scores = K.gather(scores, nms_indices)
	boxes = K.gather(boxes, nms_indices)
	classes = K.gather(classes, nms_indices)
	
	return scores, boxes, classes
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
	"""
	Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
	
	Arguments:
	yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
					box_confidence: tensor of shape (None, 19, 19, 5, 1)
					box_xy: tensor of shape (None, 19, 19, 5, 2)
					box_wh: tensor of shape (None, 19, 19, 5, 2)
					box_class_probs: tensor of shape (None, 19, 19, 5, 80)
	image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
	max_boxes -- integer, maximum number of predicted boxes you'd like
	score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
	iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
	
	Returns:
	scores -- tensor of shape (None, ), predicted score for each box
	boxes -- tensor of shape (None, 4), predicted box coordinates
	classes -- tensor of shape (None,), predicted class for each box
	"""
	# Retrieve outputs of the YOLO model
	box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

	# Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
	boxes = yolo_boxes_to_corners(box_xy, box_wh)

	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
	
	# Scale boxes back to original image shape.
	boxes = scale_boxes(boxes, image_shape)

	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
	
	return scores, boxes, classes
#########
# RNN (Recurrent Nueral Networks)
#########
def rnn_cell_forward(xt, a_prev, parameters):
	"""
	Implements a single forward step of the RNN-cell as described in Figure (2)

	Arguments:
	xt -- your input data at timestep "t", numpy array of shape (n_x, m).
	a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
	parameters -- python dictionary containing:
						Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
						Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
						Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
						ba --  Bias, numpy array of shape (n_a, 1)
						by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
	Returns:
	a_next -- next hidden state, of shape (n_a, m)
	yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
	cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
	"""
	Wax = parameters["Wax"]
	Waa = parameters["Waa"]
	Wya = parameters["Wya"]
	ba = parameters["ba"]
	by = parameters["by"]

	a_next = np.tanh(np.dot(Waa, a_prev)+np.dot(Wax, xt)+ba)
	yt_pred = softmax(np.dot(Wya, a_next)+by)
	
	cache = (a_next, a_prev, xt, parameters)
	
	return a_next, yt_pred, cache
def rnn_forward(x, a0, parameters):
	"""
	Implement the forward propagation of the recurrent neural network described in Figure (3).

	Arguments:
	x -- Input data for every time-step, of shape (n_x, m, T_x).
	a0 -- Initial hidden state, of shape (n_a, m)
	parameters -- python dictionary containing:
						Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
						Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
						Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
						ba --  Bias numpy array of shape (n_a, 1)
						by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

	Returns:
	a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
	y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
	caches -- tuple of values needed for the backward pass, contains (list of caches, x)
	"""
	caches = []
	
	n_x, m, T_x = x.shape
	n_y, n_a = parameters["Wya"].shape
	
	a = np.zeros((n_a, m, T_x))
	y_pred = np.zeros((n_y, m, T_x))
	a_next = a0
	
	# loop over all time-steps of the input 'x'
	for t in range(T_x):
		xt = x[:,:,t]
		a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
		a[:,:,t] = a_next
		y_pred[:,:,t] = yt_pred
		caches.append(cache)
	
	# store values needed for backward propagation in cache
	caches = (caches, x)
	
	return a, y_pred, caches
#LSTM
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
	"""
	Implement a single forward step of the LSTM-cell as described in Figure (4)

	Arguments:
	xt -- your input data at timestep "t", numpy array of shape (n_x, m).
	a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
	c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
	parameters -- python dictionary containing:
						Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
						bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
						Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
						bi -- Bias of the update gate, numpy array of shape (n_a, 1)
						Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
						bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
						Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
						bo --  Bias of the output gate, numpy array of shape (n_a, 1)
						Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
						by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
						
	Returns:
	a_next -- next hidden state, of shape (n_a, m)
	c_next -- next memory state, of shape (n_a, m)
	yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
	cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
	
	Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
		  c stands for the cell state (memory)
	"""
	Wf = parameters["Wf"] # forget gate weight
	bf = parameters["bf"]
	Wi = parameters["Wi"] # update gate weight (notice the variable name)
	bi = parameters["bi"] # (notice the variable name)
	Wc = parameters["Wc"] # candidate value weight
	bc = parameters["bc"]
	Wo = parameters["Wo"] # output gate weight
	bo = parameters["bo"]
	Wy = parameters["Wy"] # prediction weight
	by = parameters["by"]
	
	# Retrieve dimensions from shapes of xt and Wy
	n_x, m = xt.shape
	n_y, n_a = Wy.shape

	concat = np.concatenate((a_prev, xt), axis=0)

	ft = sigmoid(np.dot(Wf, concat) + bf)		# forget gate
	it = sigmoid(np.dot(Wi, concat) + bi)		# update gate
	cct = np.tanh(np.dot(Wc, concat) + bc)		# candidate value
	c_next = ft * c_prev + it * cct				# cell state
	ot = sigmoid(np.dot(Wo, concat) + bo)		# output gate
	a_next = ot * np.tanh(c_next)				# hidden state
	
	yt_pred = softmax(np.dot(Wy, a_next) + by)

	# store values needed for backward propagation in cache
	cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

	return a_next, c_next, yt_pred, cache
def lstm_forward(x, a0, parameters):
	"""
	Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

	Arguments:
	x -- Input data for every time-step, of shape (n_x, m, T_x).
	a0 -- Initial hidden state, of shape (n_a, m)
	parameters -- python dictionary containing:
						Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
						bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
						Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
						bi -- Bias of the update gate, numpy array of shape (n_a, 1)
						Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
						bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
						Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
						bo -- Bias of the output gate, numpy array of shape (n_a, 1)
						Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
						by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
						
	Returns:
	a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
	y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
	c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
	caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
	"""
	caches = []
	
	Wy = parameters['Wy']
	
	n_x, m, T_x = x.shape
	n_y, n_a = Wy.shape
	
	a = np.zeros((n_a, m, T_x))
	c = np.zeros((n_a, m, T_x))
	y = np.zeros((n_y, m, T_x))
	
	a_next = a0
	c_next = np.zeros(a_next.shape)
	
	for t in range(T_x):
		xt = x[:,:,t]
		a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
		a[:,:,t] = a_next
		c[:,:,t]  = yt
		y[:,:,t] = c_next
		caches.append(cache)
		
	# store values needed for backward propagation in cache
	caches = (caches, x)

	return a, y, c, caches
#Gradient Clipping
def clip(gradients, maxValue):
	'''
	Clips the gradients' values between minimum and maximum.
	
	Arguments:
	gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
	maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
	
	Returns: 
	gradients -- a dictionary with the clipped gradients.
	'''
	for gradient in gradients:
		np.clip(gradient, -maxValue, maxValue, out=gradient)
		
	return gradients
#########
# Deep learning with Tensorflow.keras
#########
import tensorflow as tf
#Callback to stop training once a certain loss has been reached
class stopCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('acc')>0.99):
			print(r"Reached 99% accuracy. Cancel training.")
			self.model.stop_training = True
#Define model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
									tf.keras.layers.Dense(512, activation=tf.nn.relu),
									tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, callbacks=[stopCallback()])
#Convolutional Model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
									tf.keras.layers.MaxPooling2D(2, 2),
									tf.keras.layers.Flatten(),
									tf.keras.layers.Dense(128, activation='relu'),
									tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=[stopCallback()])
#Save and load models
model.save('keras_model.h5')
model = tf.keras.models.load_model('keras_model.h5')
#Callback for saving the model
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('keras_model.h5', save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
#########
# Deep learning with Pytorch (extracted from the Udacity Secure and Private AI course)
#########
import torch
torch.__version__

def sigmoid_activation(x):
	""" Sigmoid activation function #https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg
	
		Arguments
		---------
		x: torch.Tensor
	"""
	return 1/(1+torch.exp(-x))

def softmax_activation(x):
	return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable
# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

y = activation(torch.sum(features * weights) + bias)
y = activation(torch.mm(features,weights.reshape(5,1))+bias) #Also can use view instead of reshape

#Higher dimension
### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1)
y = activation(torch.mm(h, W2) + B2)

#Convert from torch to numpy
import numpy as np
a = np.random.rand(4,3)
b = torch.from_numpy(a)
b.numpy()

#Building Networks with Pytorch
from torch import nn

class Network(nn.Module):
	def __init__(self):
		super().__init__()

		# Inputs to hidden layer linear transformation
		self.hidden = nn.Linear(784, 256)
		# Output layer, 10 units - one for each digit
		self.output = nn.Linear(256, 10)

		# Define sigmoid activation and softmax output 
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, x):
		# Pass the input tensor through each of our operations
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.output(x)
		x = self.softmax(x)

		return x

model = Network()

import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self):
		super().__init__()
		# Inputs to hidden layer linear transformation
		self.hidden = nn.Linear(784, 256)
		# Output layer, 10 units - one for each digit
		self.output = nn.Linear(256, 10)
		
	def forward(self, x):
		# Hidden layer with sigmoid activation
		x = F.sigmoid(self.hidden(x))
		# Output layer with softmax activation
		x = F.softmax(self.output(x), dim=1)
		
		return x

#Initializing weights and biases
model.hidden.bias.data.fill_(0)
model.hidden.weight.data.normal_(std=0.01)

#Now that we have a network, let's see what happens when we pass in an image.
# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

#PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through operations, `nn.Sequential` ([documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Sequential)). Using this to build the equivalent network:
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
					  nn.ReLU(),
					  nn.Linear(hidden_sizes[0], hidden_sizes[1]),
					  nn.ReLU(),
					  nn.Linear(hidden_sizes[1], output_size),
					  nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0, :])
helper.view_classify(images[0].view(1, 28, 28), ps)

#The operations are available by passing in the appropriate index. For example, if you want to get first Linear operation and look at the weights, you'd use model[0].
print(model[0])
model[0].weight

# Define the loss
criterion = nn.CrossEntropyLoss()
# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)
print(loss)

#########
# Train a model
#########
model = nn.Sequential(nn.Linear(784, 128),
					  nn.ReLU(),
					  nn.Linear(128, 64),
					  nn.ReLU(),
					  nn.Linear(64, 10),
					  nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		# Flatten MNIST images into a 784 long vector
		images = images.view(images.shape[0], -1)
	
		# TODO: Training pass
		optimizer.zero_grad()
		
		output = model(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
	else:
		print(f"Training loss: {running_loss/len(trainloader)}")

#########
# Validation Loop
#########
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		
		optimizer.zero_grad()
		
		log_ps = model(images)
		loss = criterion(log_ps, labels)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		
	else:
		## TODO: Implement the validation pass and print out the validation accuracy
		print(f'Accuracy: {accuracy.item()*100}%')
		
		test_loss = 0
		accuracy = 0
		
		# Turn off gradients for validation, saves memory and computations
		with torch.no_grad():
			for images, labels in testloader:
				log_ps = model(images)
				test_loss += criterion(log_ps, labels)
				
				ps = torch.exp(log_ps)
				top_p, top_class = ps.topk(1, dim=1)
				equals = top_class == labels.view(*top_class.shape)
				accuracy += torch.mean(equals.type(torch.FloatTensor))
				
		train_losses.append(running_loss/len(trainloader))
		test_losses.append(test_loss/len(testloader))

		print("Epoch: {}/{}.. ".format(e+1, epochs),
			  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
			  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
			  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

#########
# Dropout for avoiding overfitting
#########
class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 10)

		# Dropout module with 0.2 drop probability
		self.dropout = nn.Dropout(p=0.2)

	def forward(self, x):
		# make sure input tensor is flattened
		x = x.view(x.shape[0], -1)

		# Now with dropout
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = self.dropout(F.relu(self.fc3(x)))

		# output so no dropout here
		x = F.log_softmax(self.fc4(x), dim=1)

		return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		
		optimizer.zero_grad()
		
		log_ps = model(images)
		loss = criterion(log_ps, labels)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		
	else:
		test_loss = 0
		accuracy = 0
		
		# Turn off gradients for validation, saves memory and computations
		with torch.no_grad():
			model.eval()
			for images, labels in testloader:
				log_ps = model(images)
				test_loss += criterion(log_ps, labels)
				
				ps = torch.exp(log_ps)
				top_p, top_class = ps.topk(1, dim=1)
				equals = top_class == labels.view(*top_class.shape)
				accuracy += torch.mean(equals.type(torch.FloatTensor))
		
		model.train()
		
		train_losses.append(running_loss/len(trainloader))
		test_losses.append(test_loss/len(testloader))

		print("Epoch: {}/{}.. ".format(e+1, epochs),
			  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
			  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
			  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

#########
# Save and load pytorch models
#########
#Save
torch.save(model.state_dict(), 'checkpoint.pth')
#Load
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict)

######Load image data with Pytorch
import torch
from torchvision import datasets, transforms
import helper

data_dir = 'Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
								transforms.CenterCrop(224),
								transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)

data_dir = 'Cat_Dog_data'
# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
									   transforms.RandomResizedCrop(224),
									   transforms.RandomHorizontalFlip(),
									   transforms.ToTensor()]) 

test_transforms = transforms.Compose([transforms.Resize(255),
									  transforms.CenterCrop(224),
									  transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# change this to the trainloader or testloader 
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
	ax = axes[ii]
	helper.imshow(images[ii], ax=ax, normalize=False)

#########
# Transfer learning with Pytorch
#########
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'Cat_Dog_data'

#Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
									   transforms.RandomResizedCrop(224),
									   transforms.RandomHorizontalFlip(),
									   transforms.ToTensor(),
									   transforms.Normalize([0.485, 0.456, 0.406],
															[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
									  transforms.CenterCrop(224),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406],
														   [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

#Load a pretrained model
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
	param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
						  ('fc1', nn.Linear(1024, 500)),
						  ('relu', nn.ReLU()),
						  ('fc2', nn.Linear(500, 2)),
						  ('output', nn.LogSoftmax(dim=1))
						  ]))
	
model.classifier = classifier

epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
	for inputs, labels in trainloader:
		steps += 1
		# Move input and label tensors to the default device
		inputs, labels = inputs.to(device), labels.to(device)
		
		optimizer.zero_grad()
		
		logps = model.forward(inputs)
		loss = criterion(logps, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		
		if steps % print_every == 0:
			test_loss = 0
			accuracy = 0
			model.eval()
			with torch.no_grad():
				for inputs, labels in testloader:
					inputs, labels = inputs.to(device), labels.to(device)
					logps = model.forward(inputs)
					batch_loss = criterion(logps, labels)
					
					test_loss += batch_loss.item()
					
					# Calculate accuracy
					ps = torch.exp(logps)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class == labels.view(*top_class.shape)
					accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
					
			print(f"Epoch {epoch+1}/{epochs}.. "
				  f"Train loss: {running_loss/print_every:.3f}.. "
				  f"Test loss: {test_loss/len(testloader):.3f}.. "
				  f"Test accuracy: {accuracy/len(testloader):.3f}")
			running_loss = 0
			model.train()

#NLP
#Bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=3, ngram_range=(1,3))#min_df: delete words that don't appear in at least min_df documents. ngram_range: Use n-grams instead of single words, (min length, max length of ngram).
vect.fit(train_data) #train_data should be a list of sentences, paragraphs or texts
vect.vocabulary_
bag_of_words = vect.transform(train_data)
#Stop-words: Delete very frequent words, two ways of doing it:
#1 Use a language-specific list of words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
vect = CountVectorizer(min_df=3, stop_words='english')
#2 Delete words that appear very frequently.
vect = CountVectorizer(max_df=100)
#TF-IDF (term frequency - inverse document frequency): Give high weight to a word that appears frequently in a specific document but not in many documents
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=3) #Important to apply the same transformation to train and test set.
#Lemmatization: Join words with similar meanings coming from the same words. Example: plurals (car / cars), verbs (do / done / did)
import spacy
import nltk
en_nlp = spacy.load('en')
stemmer = nltk.stem.PorterStemmer()
#Topic Modeling and Document Clustering: LDA (Latent Dirichlet Allocation)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=5)
doc_topics = lda.fit_transform(X_train)
lda.components_
#Using Spacy
import spacy
nlp = spacy.load('en')
doc = nlp('This sentence belongs to the document')
#Tokenizetion:
for token in doc:
	print(token)
	print(token.lemma_) #The lemma is the base where the word comes from. Example: "walking" comes from "walk"
	print(token.is_stop) #True/False telling if it's a stopword (very frequent words that don't contain much information, example: "the", "is", "and")
#Find exact matches
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", None, *patterns) #Name for the rules, Action to take on matched words, token list where the matches appear
text_doc = nlp('Long text where the previous terms appear sometime like Galaxy Note here.')
matches = matcher(text_doc)
print(matches) #List of tuples of length 3. First element: match id, Second element: position of start, Third element: position of end.
for match in matches:
	print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")
#Categorize text. Create a pipe with Sapcy
nlp = spacy.load('en')
textcat = nlp.create_pipe('textcat', config={'exclussive_classes': True, 'architecture':'bow'})
nlp.add_pipe(textcat)
textcat.add_label('label_1')
textcat.add_label('label_2')
train_texts = df['text'].values
train_labels = [{'cats': {'label_1': label == 'label_1', 'label_2': label == 'label_2'}} for label in df['label']]
train_data = list(zip(train_texts, train_labels))
from spacy.util import minibatch
import random
optimizer = nlp.begin_training()
losses = {}
epochs = 10
for epoch in range(epochs):
	random.shuffle(train_data)
	batches = minibatch(train_data, size=8)
	for batch in batches:
		texts, labels = zip(*batch)
		nlp.update(texts, labels, sgd=optimizer, losses=losses)
	print(losses)
docs = [nlp.tokenizer(text) for text in X_val['text'].values]
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)
print(scores)
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])
#Word Embeddings from Word2Vec
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes(): #Disables all other pipes we don't need to speed things up
    vectors = np.array([token.vector for token in nlp(text)])
with nlp.disable_pipes(): #Compute vector for the whole document
    doc_vectors = np.array([nlp(text).vector for text in spam.text])
#Document similarity with cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(X, Y) #X and Y are the two vector embeddings we are looking how similar they are.


#GeoPandas
import geopandas as gpd
gdf = gpd.read_file('data.shp')
gdf.head()
gdf.plot()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10,10), color='none', edgecolor='black', zorder=3)
gdf.plot(color='blue', ax=ax)

#Coordinate Reference System (CRS)
df = pd.read_csv("data.csv")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
gdf.crs = {'init': 'epsg:4326'}

#Re-projecting (Changing the CRS)
gdf = gdf.to_crs(epsg=32630)

#Attributes of geometry objects
X = gdf.geometry.x #For points
y = gdf.geometry.y #For points
l = gdf.geometry.length #For lines
a = gdf.geometry.area #For polygons

#Get paths from a list of points
path_df = df.groupby("identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)
path_gdf.crs = {'init' :'epsg:4326'}

#Interactive Maps with Folium
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster

#function for dsiplaying maps in all browsers
def embed_map(m, file_name):
	from IPython.display import IFrame
	m.save(file_name)
	return IFrame(file_name, width='100%', height='500px')

#Create a map
m_1 = folium.Map(location=[42.32,-71.0589], tiles='openstreetmap', zoom_start=10)
#Display the map
embed_map(m_1, 'm_1.html')

#Show markers in a map
m_2 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)
#Add points to the map
for idx, row in df.iterrows():
	Marker([row['Lat'], row['Long']]).add_to(m_2)
#Display the map
embed_map(m_2, 'm_2.html')

#MarkerCluster for a lot of markers in the map
m_3 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)
#Add points to the map
mc = MarkerCluster()
for idx, row in df.iterrows():
	if not math.isnan(row['Long']) and not math.isnan(row['Lat']):
		mc.add_child(Marker([row['Lat'], row['Long']]))
m_3.add_child(mc)
#Display the map
embed_map(m_3, 'm_3.html')

#Bubble map. Circles in the map with varying size and color to show relationship between location and two other variables.
m_4 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)
#Decide colour according to the value of a variable
def color_producer(val):
	if val <= 12:
		return 'forestgreen'
	else:
		return 'darkred'
#Add a bubble map to the base map
for i in range(0,len(df)):
	Circle(
		location=[df.iloc[i]['Lat'], df.iloc[i]['Long']],
		radius=20,
		color=color_producer(df.iloc[i]['HOUR'])).add_to(m_4)
#Display the map
embed_map(m_4, 'm_4.html')

#Heatmap. Shows density of occurrence in the map
m_5 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)
#Add a heatmap to the base map
HeatMap(data=df[['Lat', 'Long']], radius=10).add_to(m_5)
# Display the map
embed_map(m_5, 'm_5.html')

#GeoCoding: Given the name of a place, get its coordinates
from geopandas.tools import geocode
result = geocode("The Great Pyramid of Giza", provider="nominatim") #Nominatim from OpenStreetMap. Returns 2 columns, geometry and address.
latitude = result['geometry'].iloc[0].y
longitude = result['geometry'].iloc[0].x
#Geocode every row in a DataFrame
def my_geocoder(row):
	try:
		point = geocode(row, provider='nominatim')['geometry'].iloc[0]
		return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})
	except:
		return None

df[['Latitude', 'Longitude', 'geometry']] = df.apply(lambda x: my_geocoder(x['Name']), axis=1)
print("{}% of addresses were geocoded!".format(
	(1 - sum(np.isnan(df["Latitude"])) / len(df)) * 100))
#Drop universities that were not successfully geocoded
df = df.loc[~np.isnan(df["Latitude"])]
df = gpd.GeoDataFrame(df, geometry=df['geometry'])
df.crs = {'init': 'epsg:4326'}

#Join geodataframes
gdf_joined = gdf.merge(df, on='name')
#spatial join
gdf_joined = gpd.sjoin(gdf1, gdf2) #looks at the geometry column in each GeoDataFrame.

#Proximity analysis
#Measuring distance between points. First make sure they use the same coordinate reference system (CRS). Also check units of the CRS.
print(gdf1.crs)
print(gdf2.crs)
gdf1_row0 = gdf1.iloc[0]
distances = gdf2['geometry'].distance(gdf1_row0['geometry'])

#Create a buffer. Get all points within a certain radius from a point.
gdf2['geometry'].buffer(2000)
