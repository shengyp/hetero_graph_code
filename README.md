# A Semantic Path Guided Model for Heterogeneous Information Network Embedding


This repository contains the code to train and evaluate models from the paper:  
_A Semantic Path Guided Model for Heterogeneous Information Network Embedding_

Clone it using:

```shell
https://github.com/shengyp/hetero_graph_code/archive/master.zip
```

<!-- 
If you find this code useful, please consider citing:


```
@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for },
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```
-->



## Contents

1. [Requirements](##Requirements)
2. [SimGAT](##SimGAT)
3. [dvantage](##dvantage)
4. [Overview](##Overview)
5. [Dataset](##Dataset)
6. [How_to_build_semantic_path_and_simpath_matrix](##How to build semantic path and simpath matrix?)
7. [Run](##Run)
8. [Training](##Training)
9. [Testing](##Testing)
10. [Visualization](##Visualization)
11. [Contact](##Contact)

## Requirements

 - python 3.7.0
 - numpy 1.16.2
 - tensorflow 1.13.1

## SimGAT
The source code of Attention Guided Simpath Matrix Construction forHeterogeneous Information Network Embedding
The source code is based on [HAN](https://github.com/Jhy1993/HAN)


## Advantage
1. We use the Simpath similarity matrix in a heterogeneous network as the importance weight between neighboring nodes, which is more interpretable than training with a neural network.
2. After using the Simpath matrix as the importance weight between neighboring nodes, good results can be achieved without the need for multiple heads of attention, which reduces the time required for the model to run and improves the operating efficiency.
3. Different from HAN, we have optimized the calculation formula of the semantic path weights. In the calculation process, it is not necessary to specify the dimensions of the transformation matrix, which can better obtain the importance of graphs constructed by different semantic paths
4. After using our model to train the weights between semantic paths, if we get new data, use the new data to build a semantic path adjacency matrix, and linearly combine the results with the obtained weights as final embedding for classification and clustering tasks also can get good results. (Note: the new data has not been involved in training)


## Overview
Here we provide the implementation of a SimGAT layer in TensorFlow, along with a minimal execution example (on the DBLP dataset). The repository is organised as follows:  
 - models/ contains the implementation of the SimGAT network (gat.py);
 - utils/ contains:  
     - Similarity matrix is used as node weight to calculate(layers.py);
     - preprocessing subroutines (process.py);
 - visualization/ Clustering visualization results(Cluster_visualization.ipynb);
 - premodel/ Model and final embedding save location.


## Dataset
Preprocessed DBLP can be found in https://pan.baidu.com/s/1J15d3egtnnjaYcPg22iIww , 提取码：g4df  
HUAWEI dataset can be found in https://pan.baidu.com/s/1cg_xqSbPZrPlY_ekM9WMNA , 提取码：kpq8  
IMDB dataset can be found in https://pan.baidu.com/s/1aSTzCsCezW8zd9E0e6KPGg , 提取码：nmpe  
Douban dataset can be found in https://pan.baidu.com/s/1S_tywZran1mQR1kKQdwNjw , 提取码：xwnt  


## How to build semantic path and simpath matrix?
Demo: ```data_deal.py ``` and ```meta-pathCreate.py ```


## Run
Download preprocessed data and modify data read path,
Run the program file ```simGAT_DBLP4057.py ```


## Training
 - Train the model with:
 ```- dataset /dataDBLP/features.npy, one_hot_labels.npy, small_adj_data.npy, test_idx.npy, train_idx.npy
    - lr = 0.003 
    - nonlinearity = tf.nn.elu
    - l2_coef = 0.001
 ```
 - During the training process, a feature, adj_matrix and sim_matrix will be allocated for each semantic path, which may increase your memory.
 - If you use a server, you can specify the serial number of the GPU, which can get training results faster.
 - If the parameter passed in is different from the default parameter value, training will be given priority over the parameter passed in.
  

## Testing
 -  Extract features from test set. They will be saved in ```./premodel ```.
 - After feature extraction, call the function in ```jhyexps.py ``` to calculate the evaluation index of classification (F1-micro, Fi-macro) and clustering (NMI, ARI, FMI).
 - There is no need to deal with the above, you can directly get the running result of the evaluation index by calling the running function.


## Visualization

We provide a script to visualize the clustering effect of the final embedding in ```./visualization/Cluster_visualization.ipynb ```. And provided some samples for the program to run: test1.npy and test3.npy


## Contact

For any questions or suggestions you can use the issues section or reach us at shengyp2011@163.com
