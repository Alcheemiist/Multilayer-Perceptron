import sys
from data import extract_csv, preprocess_binary_output_data 
import nnmodule as nn
import numpy as np

def read_data(path="../data"):
    with open(f"{path}/train_X.npy", 'rb') as f:
        train_X = np.load(f)
    with open(f"{path}/train_Y.npy", 'rb') as f:
        train_Y = np.load(f)
    with open(f"{path}/test_X.npy", 'rb') as f:
        test_X = np.load(f)
    with open(f"{path}/test_Y.npy", 'rb') as f:
        test_Y = np.load(f)
    return test_X, test_Y, train_X, train_Y

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) == 2 else exit("No data path provided")
    dataset = (train_X, train_Y), (test_X, test_Y) = preprocess_binary_output_data(extract_csv(path))

    print(f"train shape : {train_X.shape}, {train_Y.shape}, test shape : {test_X.shape}, {test_Y.shape}")
	
    model_path = "../model/model.json"
    model_data = nn.parse_model_json(model_path)
    model = nn.parse_model.compile_and_fit_parsed_model(model_data, preprocess_func=None, data=dataset)

    # save the model
    model.save("../model/model.npy")
    model.evaluate(test_X, test_Y)