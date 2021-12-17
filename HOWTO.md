# Setup

## Requirements

* Python3.8 
* [PyTorch (1.10.1)](https://pytorch.org/), preferable with CUDA the Compute Platform.
* Other dependencies are specified in: `pip install -r requirements.txt`

## Minimal working example:

To test if everything works correctly, train a very small number of batches.

```
python train.py data/actions/ logs/actions --num-batches=10
```

This should take no more than a minute. To view the training result, you can use `tensorboard`.
```
tensorboard --logdir logs/actions --bind_all
```

This should open a webbrowser with the training experiment.