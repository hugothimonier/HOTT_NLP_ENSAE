import pandas as pd
import os
import glob
import numpy as np 
os.chdir('/Users/Hugo/Documents/Github/HOTT_NLP_ENSAE/NYT_dataset')
cols = ['commentBody','sectionName', 'newDesk']

files = glob.glob('*.csv')
for file in files :
	df = pd.read_csv(file, encoding = 'latin')
	df = df[cols]
	df_split = np.array_split(df, 2)
	df_split[0].to_csv(file.replace('.csv','_1.csv'))
	df_split[1].to_csv(file.replace('.csv','_2.csv'))

	