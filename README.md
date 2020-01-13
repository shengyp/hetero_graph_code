# Attention Guided Simpath Matrix Construction for Heterogeneous Information Network Embedding


This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_


Clone it using:

```shell
git clone --recursive https://github.com/torralba-lab/im2recipe.git
```


If you find this code useful, please consider citing:

```
@inproceedings{salvador2017learning,
  title={Attention Guided Simpath Matrix Construction forHeterogeneous Information Network Embedding},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Contents
1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-dataset)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Prepare training data](#prepare-training-data)
6. [Training](#training)
7. [Testing](#testing)
8. [Visualization](#visualization)
9. [Pretrained model](#pretrained-model)
10. [Contact](#contact)


## Installation



## SimGAT
The source code of Attention Guided Simpath Matrix Construction forHeterogeneous Information Network Embedding
The source code is based on [HAN](https://github.com/Jhy1993/HAN)

## Advantage
1. We use the Simpath similarity matrix in a heterogeneous network as the importance weight between neighboring nodes, which is more interpretable than training with a neural network.
2. After using the Simpath matrix as the importance weight between neighboring nodes, good results can be achieved without the need for multiple heads of attention, which reduces the time required for the model to run and improves the operating efficiency.
3. Different from HAN, we have optimized the calculation formula of the semantic path weights. In the calculation process, it is not necessary to specify the dimensions of the transformation matrix, which can better obtain the importance of graphs constructed by different semantic paths
4. After using our model to train the weights between semantic paths, if we get new data, use the new data to build a semantic path adjacency matrix, and linearly combine the results with the obtained weights as final embedding for classification and clustering tasks also can get good results. (Note: the new data has not been involved in training)

## Dataset
Preprocessed DBLP can be found in https://pan.baidu.com/s/1J15d3egtnnjaYcPg22iIww , 提取码：g4df  
HUAWEI dataset can be found in https://pan.baidu.com/s/1cg_xqSbPZrPlY_ekM9WMNA , 提取码：kpq8  
IMDB dataset can be found in https://pan.baidu.com/s/1aSTzCsCezW8zd9E0e6KPGg , 提取码：nmpe  
Douban dataset can be found in https://pan.baidu.com/s/1S_tywZran1mQR1kKQdwNjw , 提取码：xwnt  

## How to build semantic path and simpath matrix?
Demo:Graph.py and Sim.py

## Run
Download preprocessed data and modify data read path,
Run the program file ```simGAT_DBLP4057.py ```

## Training

## Testing
 -  Extract features from test set. They will be saved in ```./premodel ```.
 - After feature extraction, call the function in ```jhyexps.py ``` to calculate the evaluation index of classification (F1-micro, Fi-macro) and clustering (NMI, ARI, FMI).
 - There is no need to deal with the above, you can directly get the running result of the evaluation index by calling the running function.

## Visualization

We provide a script to visualize the clustering effect of the final embedding in ```./visualization/Cluster_visualization.ipynb ```. And provided some samples for the program to run: test1.npy and test3.npy

## Contact

For any questions or suggestions you can use the issues section or reach us at shengyp2011@163.com
