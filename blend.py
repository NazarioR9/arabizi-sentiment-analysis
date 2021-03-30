import os, sys, glob
import numpy as np
import pandas as pd

FOLDER = 'submissions/'
LABEL_COL = 'label'

def literal_eval(x):
	return list(map(float, x[1:-1].split()))

def retrieve_file(names):
	return [glob.glob(f'{FOLDER}*{name}*_raw_outputs.csv')[0] for name in names]

def main():
	names = ['rzA27Luehf', 'AH7LwUXCvT', '10WwJdQcXs']
	weights = [0.44, 0.36, 0.2]

	#Blending
	csvs = [pd.read_csv(csv) for csv in retrieve_file(names)]
	avg = np.average([csv[LABEL_COL].apply(literal_eval).values.tolist() for csv in csvs], weights=weights, axis=0)

	blend = csvs[0].copy()
	blend[LABEL_COL] = np.argmax(avg, axis=1) - 1

	csv_name = f'{FOLDER}final_submission.csv'

	blend.reset_index(drop=True, inplace=True)
	blend.to_csv(csv_name, index=False)

	#Sanity check
	print(csv_name)
	print(blend.head())








if __name__ == '__main__':
	main()
