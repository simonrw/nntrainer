# Neural Network Trainer

Quickly and interactively train a neural network image classifier.

## Key inputs required from the user

* network architecture (e.g. `VGG16`, `Resnet50`)
* number of output classes
* hyperparameters?
* training algorithm (`Adam`, `SGD`?)

## Implementation ideas

### Desktop GUI

Use PyQt to build a local GUI. This has the advantage of being runable from the local machine with the data on, however requires being run from the machine with the GPU (i.e. the same machine).

### Web app

**abandoned**

This is hostable, so can be run on the GPU machine, however requires the data to be sent to the server somehow. Uploading a zip file is inconvenient, but may be the best approach.

This also allows the user to "submit a training job" and come back to it later, rather than leaving the GUI running.
