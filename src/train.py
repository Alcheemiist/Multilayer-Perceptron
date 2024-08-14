import sys
from data import extract_csv, preprocess_binary_output_data 
import nnmodule as nn

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