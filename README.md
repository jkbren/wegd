# wegd: The within-ensemble graph distance
Python code for calculating *within-ensemble graph distance* between networks
sampled iidly from a given graph ensemble, under a number of graph distances.

This code accompanies the recent paper: 

**The within-ensemble graph distance**\
Harrison Hartle, Brennan Klein, Stefan McCabe, Guillaume St-Onge,
Charles Murphy, Alexander Daniels, and Laurent Hébert-Dufresne.

- - - -

<p align="center">
<img src="figs/pngs/GraphWEnD_example.png" alt="example RGG ER" width="95%"/>
</p>

**<p align="center">Fig. 1: Mean within-ensemble distances for each distance measure tested.**

## Analysis and Notebooks (works in progress...)

1. [pipeline](https://github.com/jkbren/GraphWEnD/blob/master/pipeline.ipynb)
2. etc.


## Requirements  <a name="requirements"/>

This code is written in [Python 3.x](https://www.python.org) and uses 
the following packages:

* [NetworkX](https://networkx.github.io)
* [Scipy](http://www.scipy.org/)
* [Numpy](http://numpy.scipy.org/)
* and for the graph distances, we use [netrd](https://github.com/netsiphd/netrd/)

## Citation   <a name="citation"/>

If you use these methods and this code in your own research, please cite our paper:

Hartle, H., Klein, B., McCabe, S., St-Onge, G., Murphy, C., Daniels, A.,
and Hébert-Dufresne, L. (2020).
**Network comparison and the within-ensemble graph distance.**

Bibtex: 
```text
@article{hartle2020wegd,
  title = {Network comparison and the within-ensemble graph distance},
  author = {Harrison Hartle and Brennan Klein and Stefan McCabe and Guillaume St. Onge and Charles Murphy and Alexander Daniels and Laurent Hébert-Dufresne},
  journal = {arXiv preprint XXX},
  year = {2020}
}
```
