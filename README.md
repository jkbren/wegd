# wegd: The within-ensemble graph distance
Python code for calculating the *within-ensemble graph distance* between networks
sampled from a given graph ensemble, for various graph distances.

This code accompanies the paper: 

**Network comparison and the within-ensemble graph distance**\
Harrison Hartle, Brennan Klein, Stefan McCabe, Alexander Daniels,
Guillaume St-Onge, Charles Murphy, and Laurent Hébert-Dufresne.
https://arxiv.org/abs/2008.02415

- - - -

<p align="center">
<img src="figs/pngs/gnp_rgg_wegd_p_n500.png" alt="example RGG ER" width="95%"/>
</p>

**<p align="center">Fig. 1: Mean within-ensemble distances for each distance measure tested.**

## Analysis and Notebooks

1. [A notebook illustrating usage of the code](https://github.com/jkbren/wegd/blob/master/code/wegd-example.ipynb)
2. [Code for replicating the paper's batch experiments on a computing cluster](https://github.com/jkbren/wegd/blob/master/cluster/)
3. [A notebook creating the figures of the paper](https://github.com/jkbren/wegd/blob/master/code/make-all-plots.ipynb) using our [replication data](https://github.com/jkbren/wegd/tree/master/data).


## Requirements  <a name="requirements"/>

This code is written in [Python 3.x](https://www.python.org) and uses 
the standard Python scientific computing stack and [netrd](https://github.com/netsiphd/netrd/)
for implementations of the graph distances.

The specific dependencies are documented in `requirements.txt`.

## Citation   <a name="citation"/>

If you use these methods and this code in your own research, please cite our paper:

Hartle H., Klein B., McCabe S., Daniels A., St-Onge G., Murphy C.,
and Hébert-Dufresne L. (2020).
**Network comparison and the within-ensemble graph distance.**
*Proc. R. Soc. A* 20190744. http://dx.doi.org/10.1098/rspa.2019.0744

Bibtex: 
```text
@article{hartle2020wegd,
  title = {Network comparison and the within-ensemble graph distance},
  author = {Harrison Hartle and Brennan Klein and Stefan McCabe and Alexander Daniels and Guillaume St. Onge and Charles Murphy and Laurent Hébert-Dufresne},
  journal = {Proceedings of the Royal Society A},
  year = {2020},
  doi = {10.1098/rspa.2019.0744}
}
```

## License

This code is available for academic use only. However, much of the work is done by the
netrd library, which is available under an MIT license.
