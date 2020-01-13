# Attention Guided Simpath Matrix Construction forHeterogeneous Information Network Embedding

## SimGAT
The source code of Attention Guided Simpath Matrix Construction forHeterogeneous Information Network Embedding
The source code is based on [HAN](https://github.com/Jhy1993/HAN)

## How to build semantic path and simpath matrix?
Demo:Graph.py and Sim.py

## Dataset
Preprocessed DBLP can be found in https://pan.baidu.com/s/1J15d3egtnnjaYcPg22iIww, 提取码：g4df

## Vision models

We used the following pretrained vision models:

- VGG-16 ([prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt) and [caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)).

when training, point arguments ```-proto``` and ```-caffemodel``` to the files you just downloaded.

- ResNet-50 ([torchfile](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7)).

when training, point the argument ```-resnet_model``` to this file.

## Out-of-the-box training

To train the model, you will need the following files:
* `data/data.h5`: HDF5 file containing skip-instructions vectors, ingredient ids, categories and preprocessed images.
* `data/text/vocab.bin`: ingredient Word2Vec vocabulary. Used during training to select word2vec vectors given ingredient ids.

The links to download them are available [here](http://im2recipe.csail.mit.edu/dataset/download).

## Prepare training data

We also provide the steps to format and prepare Recipe1M data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

### Choosing semantic categories

We provide the script we used to extract semantic categories from bigrams in recipe titles:

- Run ```python bigrams --crtbgrs```. This will save to disk all bigrams in the corpus of all recipe titles in the training set, sorted by frequency.
- Running the same script with ```--nocrtbgrs``` will create class labels from those bigrams adding food101 categories.

These steps will create a file called ```classes1M.pkl``` in ```./data/``` that will be used later to create the HDF5 file including categories.




## Visualization

We provide a script to visualize top-1 im2recipe examples in ```./pyscripts/vis.py ```. It will save figures under ```./data/figs/```.

## Pretrained model

Our best model can be downloaded [here](http://data.csail.mit.edu/im2recipe/im2recipe_model.t7.gz).
You can test it with:
```
th main.lua -test 1 -loadsnap im2recipe_model.t7
```

## Contact

For any questions or suggestions you can use the issues section or reach us at amaia.salvador@upc.edu or nhynes@mit.edu.
