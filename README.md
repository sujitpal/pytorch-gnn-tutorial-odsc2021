# Deep Learning with Graphs -- an introduction to Graph Neural Networks (with code examples in Pytorch Geometric)

_Repository for tutorial to be presented at [ODSC West 2021](https://odsc.com/california)._

## Abstract

A graph is a data structure composed of nodes interconnected by edges. Many real world data can be represented by graphs, in application domains as diverse as social networks to  biochemistry. Graph Neural Networks (GNN) are a relatively new type of Deep Learning architecture that have evolved to work effectively with these data structures. Traditional network architectures such as Convolutional Neural Networks (CNN) and Recurrent Neural Networks are designed around the idea of leveraging spatial and temporal locality respectively, and are thus optimized for use in 2-d and 3-d images and sequential data such as text, audio, and time series, which exhibit these properties. GNNs, on the other hand, are designed to work with typical characteristics of graph structure, such as their complex topology and indeterminate size. GNNs are flexible enough to solve different classes of graph tasks, i.e. node level tasks such as node classification, edge level tasks such as link prediction and recommendation, and graph or subgraph level tasks such as finding graph isomorphism, etc. GNNs thus provide an efficient and scalable way to do deep learning against graph structured data and solve novel problems.

In this tutorial, we will introduce GNN concepts and popular GNN architectures such as Graph Convolution Network (GCN), GraphSAGE, and Graph Attention Network (GAT), and describe how they can be used to solve different types of graph tasks. We will demonstrate examples of different types of GNN using Pytorch and Pytorch Geometric. Pytorch is a popular library for deep learning in Python, and Pytorch Geometric is a library for doing deep learning specifically on irregular data structures such as graphs.

Attendees are expected to come away from the talk with an understanding of GNN features and tools for implementing them. To derive the most value from the talk, attendees should be familiar with Python and Pytorch programming.

## Table of Contents

* [GNN Basics](01-gnn-basics.md)
* Node classification example ([Exercise](02-node-classification-exercise.ipynb) | [Solution](02-node-classification-solution.ipynb))
* Graph classification example ([Exercise](03-graph-classification-exercise.ipynb) | [Solution](03-graph-classification-solution.ipynb))
* Link Prediction example ([Exercise](04-link-prediction-exercise.ipynb) | [Solution](04-link-prediction-solution.ipynb))
* [GNN Extensions](05-gnn-extensions.md)
* Custom Layer example ([Exercise](06-custom-layer-exercise.ipynb) | [Solution](06-custom-layer-solution.ipynb))
* Custom Dataset example ([Exercise](07-custom-dataset-exercise.ipynb) | [Solution](07-custom-dataset-solution.ipynb))

## Running the examples

* (optional) Fork the repository
* Navigate to the [Colab Web UI](https://colab.research.google.com/) 
* Click on the GitHub tab. 
* Enter the URL of your forked (or this) repository in the field titled "Enter a GitHub URL..." and hit the search icon. 
* You should see the notebooks appear in the results. Click on the notebook you want to work with in Colab.

Notebooks are numbered sequentially. To implement the examples, open the corresponding exercise notebook suffixed with `-exercise.ipynb` in the file name. They contain step-by-step instructions to help you implement the GNN or custom component. The corresponding solution notebooks are suffixed with `-solution.ipynb` in the file name, and contain my solution to these examples. For best results, try to hold off looking at them unless you are truly stuck or want to compare with your approach.

## References

* [Graph Neural Networks and Some of GNN Applications -- Everything you need to know](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications) (Menzil 2021) -- blog post
* [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
* [Graph Convolutional Networks](https://tkipf.github.io/graph-convolutional-networks/) (Kipf 2016) -- blog post
* [Pytorch Geometric Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) -- PyG documentation
