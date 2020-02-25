# Probabilistic Quantum Clustering (PQC)


### Usage
- *PQC_main_script.m* contains the main script with the pipeline of the PQC algorithm.

-Before, add project folder to Matlab search path.

- The script uses 5 datasets as examples, they are contained in *datasets4.mat*, where one of them is selected by the *opt* variable.

- There are two model variants implemented, QC2 and QC3. They can be switched with *QC3* = True/False

  - QC2 is more basic and estimates the density of the points by K-NN with spherical distributions.
  - QC3 is more complex and estimates the density of the points by K-NN with distributions based on covariance matrices.
  
- The two main hyper-parameters can be scanned enabling the following variables (*scan_knn* and *scan_dE*). The graph with average negative log-likelihood (ALL) will help to select the most appropriate hyper-parameter (the smaller ALL the better).  

*TODO: Add additional guidelines to clarify the usage*


### Additional information and mathematical background can be found in published articles:

- [Current version in Knowledge-Based Systems](https://doi.org/10.1016/j.knosys.2020.105567)

- [arXiv preliminar version](https://arxiv.org/abs/1902.05578)
