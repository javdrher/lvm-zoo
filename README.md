# lvm-zoo
LVM Zoo is a small package intended to contain some latent variable models implemented using the GPflow project.
Currently, only the partial predictions of the Bayesian GP-LVM are included. It is ongoing work and comes with no
guarantees.

# Install
1) Install Tensorflow
```
pip install tensorflow
```

2) Clone the repo and install:
```
pip install . --process-dependency-links
```
If you do not wish to install the package, rather modify the sources and test you can also run
```
python setup.py develop
```