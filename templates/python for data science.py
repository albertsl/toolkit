#Albert Sanchez Lafuente 2/4/2019, Pineda de Mar, Spain
#https://github.com/albertsl/
#Structure of the template mostly based on the Appendix B of the book Hands-on Machine Learning with Scikit-Learn and TensorFlow by Aurelien Geron (https://amzn.to/2WIfsmk)
#Load packages
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
sns.set()
import numba

#Check versions
import platform
print("Operating system:", platform.system(), platform.release())
import sys
print("Python version:", sys.version)
print("Numpy version:", np.version.version)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Numba version:", numba.__version__)

#Load data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
#If data is too big, take a sample of it
df = pd.read_csv('file.csv', nrows=50000)
#Load mat file
from scipy.io import loadmat
data = loadmat('file.mat')
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
#For a categorical dataset we want to see how many instances of each category there are
df['categorical_var'].value_counts()
#Automated data visualization
from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')

#Define Validation method
#Train and validation set split
from sklearn.model_selection import train_test_split
X = df.drop('target_var', axis=1)
y = df['column to predict']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, stratify = y.values, random_state = 101)
#Cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)
#StratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=101)
for train_index, val_index in skf.split(X, y):
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]

#Select columns of a certain type
df_bool = df.select_dtypes(include='bool')
df_noint = df.select_dtypes(exclude='int')

#Check for missing data
total_null = df.isna().sum().sort_values(ascending=False)
percent = 100*(df.isna().sum()/df.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
#Generate new features with missing data
nanf = ['1']
for feature in nanf:
    df[feature + '_nan'] = df[nanf].isna()
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
from math import isnan
for i in df.iterrows():
	if isnan(i[1][col]):
		df[col].loc[i[0]] = i*3

#Drop columns/rows
df.drop('column_full_of_nans')
df.dropna(how='any', inplace=True)

#Fix Skewed features
from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
#Box Cox Transformation of (highly) skewed features. We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x
#Note that setting λ=0 is equivalent to log1p
from scipy.special import boxcox1p
skewed_features = skewness.index
lambd = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lambd)
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
sns.pairplot(df)
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

#Feature selection: Drop attributes that provide no useful information for the task
#Unsupervised Feature selection before training a model
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X_train, y_train)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

featureScores.sort_values('Score', ascending=False) #The highest the number, the more irrelevant the variable is

#Feature engineering. Create new features by transforming the data
#Discretize continuous features
#Decompose features (categorical, date/time, etc.)
#Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.)
#Aggregate features into promising new features (x*y)
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
scaler = RobustScaler()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)

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
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.0005, random_state=101)
ls.fit(X_train,y_train)

y_pred = ls.predict(X_val)

#########
# Ridge Regression
#########
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
model = xgb.XGBClassifier(random_state=1, n_estimators=1000, learning_rate=0.01) #for the best model, high number of estimators, low learning rate
model.fit(x_train, y_train)
model.score(x_test,y_test)
#Regression
model=xgb.XGBRegressor(random_state=1, n_estimators=1000, learning_rate=0.01) #for the best model, high number of estimators, low learning rate
model.fit(x_train, y_train)
model.score(x_test,y_test)

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
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
model = CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_val, y_val))
model.score(X_val, y_val)
#Regression
model = CatBoostRegressor()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train, y_train,cat_features=categorical_features_indices,eval_set=(X_val, y_val))
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
# K-Means Clustering
#########
#Find parameter k: Elbow method
SSE = []
for k in range(1,10):
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(df)
	SSE.append(kmeans.inertia_)

plt.plot(list(range(1,10)), SSE)

#Train model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K) #Choose K
kmeans.fit(df)

#Evaluate the model
kmeans.cluster_centers_
kmeans.labels_

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

#########
# Multi-layer Perceptron Regressor (Neural Network)
#########
from sklearn.neural_network import MLPRegressor

lr = 0.01 #Learning rate
nn = [2, 16, 8, 1] #Neurons by layer

MLPr = MLPRegressor(solver='sgd', learning_rate_init=lr, hidden_layer_sizes=tuple(nn[1:]), verbose=True, n_iter_no_change=1000, batch_size = 64)
MLPr.fit(X_train, y_train)
MLPr.predict(X_val)

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
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.index = X_train.index

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(X_test_PCA)
X_test_PCA.index = X_test.index

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
