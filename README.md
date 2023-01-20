# Multiplex Network Reconstruction
This is the implementation of numerical experiments in our paper titled *Reconstructing Multiplex Networks with Application to Covert Networks*. If you find our paper or code useful, we kindly ask you to cite our work

```
@article{yu2023reconstructing,
  title={Reconstructing Sparse Multiplex Networks with Application to Covert Networks},
  author={Yu, Jin-Zhu and Wu, Mincheng and Bichler, Gisela and Aros-Vera, Felipe and Gao, Jianxi},
  journal={Entropy},
  volume={25},
  number={1},
  pages={142},
  year={2023},
  publisher={MDPI}
}
````

# Requirements
### Software
Python 3.8.3 

### Packages
numpy 1.19.2

pandas 1.4.4

numba 0.55.1

sklearn 1.1.1

imblearn 0.7.0

networkx 2.8.4

### Run
To get the results for reconstructuring each multiplex network, run ```multi_net.py``` and then ```plot_metrics.py``` with the respective ```net_name```, ```n_layer```, and ```n_node_total```.

Parellel processing is used to reduce the runtime. If necessary, Cython can be used to decrease the runtime a bit more.