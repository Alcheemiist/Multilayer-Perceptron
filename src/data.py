import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

tolerance = 0.7

class PreprocessingError(Exception):
	pass

def extract_csv(filepath: str):
    """Extracts the csv file from the given path"""
    try:
        return pd.read_csv(filepath, header=None)
    except:
        raise FileNotFoundError
    
def standardize_data(data):
	"""Standardizes the data using mean and standard deviation"""
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	standardized_data = (data - mean) / std
	return standardized_data

def convert_target_column(df: pd.DataFrame):
	"""Converts the target column to 0 and 1 and moves it to the first column"""
	
	target_columns = [col for col, val in df.nunique().items() if val ==  2]
	if len(target_columns) ==  1:
		target_column = target_columns[0]
	else:
		raise PreprocessingError("Too much target columns")
	df[0], df[target_column] = df[target_column].copy(), df[0].copy()
	le = LabelEncoder()
	df[0] = le.fit_transform(df[0])
	return df

def drop_unrelated_columns(df: pd.DataFrame):
	"""Drops columns with correlation less than tolerance"""
	for i in range(31):
		if (i != 1):
			correlation = df[0].corr(df[i])
			if (correlation < tolerance):
				df.drop(i, axis=1, inplace=True)
	label_map = {old: new for new, old in enumerate(df.columns)}
	label_map[0] = "predict"
	df = df.rename(columns=label_map)
	return df

def split_dataset(df: pd.DataFrame):
	"""Splits the dataset into training and testing data with 80% and 20% respectively"""
	df = df.sample(frac=1).reset_index(drop=True)
	
	split_percentage = 0.8
	index = int(len(df) * split_percentage)
	Y = df["predict"]
	df = df.drop("predict", axis=1)

	y = np.array(Y)
	x = np.array(df)
	indices = np.random.permutation(len(x))
	y = y[indices]
	x = x[indices]

	return (x[:index],x[index:]),(y[:index], y[index:])

def preprocess_binary_output_data(data: str | pd.DataFrame):
	"""Preprocess the data for binary output"""
	if (isinstance(data, str)):
		df = extract_csv(data)
	elif (isinstance(data, pd.DataFrame)):
		df = data
	else:
		raise ValueError("preprocess_binary_output_data(): Invalid type")

	df = convert_target_column(df)
	df = drop_unrelated_columns(df)
	without_pred_df = df.drop("predict", axis=1)
	df[df.columns[1:]] = standardize_data(without_pred_df)

	(train_X, test_X), (train_Y, test_Y) = split_dataset(df)
	if (len(test_Y) == test_X.shape[0]):
		train_Y = train_Y.reshape(-1, 1)
		test_Y = test_Y.reshape(-1, 1)
	return (train_X, train_Y), (test_X, test_Y)

if __name__ == "__main__":
    if sys.argv is None:
        raise ValueError("No file path provided")
    elif len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments")
    else:
        path = sys.argv[1]
	
    dataset = (train_X, train_Y), (test_X, test_Y) = preprocess_binary_output_data(extract_csv(path))
    print(f"train shape : {train_X.shape}, {train_Y.shape}, test shape : {test_X.shape}, {test_Y.shape}")
	
    # save the train and test dataset
    np.save("../data/train_X.npy", train_X)
    np.save("../data/train_Y.npy", train_Y)
    np.save("../data/test_X.npy", test_X)
    np.save("../data/test_Y.npy", test_Y)