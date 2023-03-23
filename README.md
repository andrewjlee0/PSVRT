# PyTorch-Integrated Code for the Parametric Synthetic Visual Reasoning Test (PSVRT) Repository

This repository contains code to run experiments used in the paper: Kim, Ricci and Serre (2018): Not-So-CLEVR: Learning same-different relations strains feedforward neural networks.

Minor portions of files throughout the repository have been rewritten to integrate with a PyTorch project. Some of those re-written files include 'psvrt.py' and 'params.py'.

In the only new file 'andrew.py', I write a new PyTorch DataSet object that uses the psvrt class to generate some input number of PSVRT images, using the default image parameters listed in 'params.py'. This DataSet object then feeds naturally into PyTorch's DataLoaders object.

The rewritten files were written precisely because this repository is unfortunately not a package. When files "call" each other, through import statements and the like, Python does not know which file or directory to look because the repository is not an independent package that can be called. To go around this issue, I first download the repository as a folder directly into my PyTorch project folder. Then, I call the relevant files I need, using import statements and from statements, to create the new PyTorch DataSet object within the file I am running tests (i.e., 'main_psvrt.py'). Finally, I run this main file, and I remove as many bug-causing import statements and from statements in the repository's files, like 'psvrt.py' and 'params.py', that the Python compiler raises to me. Eventually, things end up working!
