# Multilayer Perceptron — a neural network from scratch in NumPy

A small deep-learning framework written from scratch in NumPy, with no PyTorch or TensorFlow. It's used to classify breast-cancer biopsies (Wisconsin Diagnostic dataset, 30 features) as **malignant** or **benign**.

**Result:** 99.1% accuracy on the held-out test set (113 / 114 correct), binary cross-entropy loss 0.143.

## What's implemented

| Module (`src/nnmodule/`) | Contents |
|---|---|
| `layer.py` | Fully connected `Dense` layer: forward pass and backpropagation |
| `activation.py` | ReLU, Sigmoid, Softmax, and fused Softmax + categorical cross-entropy |
| `loss.py` | Binary and categorical cross-entropy |
| `optimizer.py` | SGD and **Adam** |
| `initializer.py` | Weight initialisers: zero, random normal, He (normal and uniform), Xavier, LeCun |
| `model.py` | Training loop, evaluation, accuracy and loss history, save/load |
| `parse_model.py` | Builds a network from a **JSON architecture file** |

The architecture is configuration, not code:

```json
{
  "loss_function": "binarycrossentropy",
  "layers": [
    {"type": "Dense", "units": 24, "activation": "relu", "weights_initializer": {"type": "heUniform"}},
    {"type": "Dense", "units": 24, "activation": "relu"},
    {"type": "Dense", "units": 2,  "activation": "sigmoid"}
  ],
  "epochs": 1001,
  "optimizer": {"type": "Adam"}
}
```

## Run it

```bash
git clone https://github.com/Alcheemiist/Multilayer-Perceptron.git
cd Multilayer-Perceptron
./setup.sh && source venv/bin/activate

cd src
python train.py ../data/data.csv     # split, train with model/model.json, save model/model.npy
python predict.py                     # evaluate the saved model on the held-out test set
python plot-all-metrics.py            # compare loss and accuracy curves across runs (historics/)
```

## Why from scratch

Writing backprop, Adam's moment estimates and numerically stable softmax + cross-entropy by hand is the fastest way to understand what frameworks do for you, and why training sometimes diverges.

---

[Elmahdi Elaazmi](https://elaazmielmahdi.com) · 1337 / 42 Network AI & ML track.
