import nnmodule as nn
import numpy as np
import matplotlib.pyplot as plt

def laod_data_test(pathX="../data/test_X.npy", pathY="../data/test_Y.npy"):
    with open(pathX, 'rb') as f:
        test_X = np.load(f)
    with open(pathY, 'rb') as f:
        test_Y = np.load(f)
    return test_X, test_Y

def load_model(model_path="../model/model.npy"):
    model = nn.Model()
    with open(model_path, 'rb') as f:
        model.layers = np.load(f, allow_pickle=True).tolist()
        model.activations = np.load(f, allow_pickle=True).tolist()
        model.optimizer = np.load(f, allow_pickle=True).item()
        model.loss = np.load(f, allow_pickle=True).item()
        model.accuracies = np.load(f, allow_pickle=True).tolist()
    print(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    model_path = "../model/model1.npy"
    test_X_path = "../data/test_X.npy"
    test_Y_path = "../data/test_Y.npy"

    test_X, test_Y = laod_data_test(test_X_path, test_Y_path)
    model = load_model(model_path)
    data =  model.evaluate(test_X, test_Y)
    # prediction = data["prediction"]
    # accuracy = data["accuracy"]
    # print(f"Accuracy: {accuracy}")
    # print(f"Predictions: {prediction}\n")
    # print(f"True values: {test_Y.tolist()}")