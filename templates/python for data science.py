# 1 Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
	# 2 Load data
	df = pd.read_csv('file.csv')
	# 3 Visualize data
	df.head()
	df.describe()

if __name__ == "__main__":
	main()
