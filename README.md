# Deep Embedded Clustering generalisability and adaptation for mixed data types
This code was used for a scientific publication where we recreated a Deep Embedded Clustering model on a populations of ICU patients. We tested the model's generalisability onto a population of ICU patients from another hospital. We Also created the X-DEC model adaptation, which uses an XVAE to integrate mixed datatypes, and tested its generalisability as well.

## Guide to using the code
See the diagram below for some guidance on how to use and set-up the code to work on your computer.

![Guide to using the code](https://github.com/DAM-IC/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types/assets/29426481/aae11d9e-b018-4145-91cc-197164493289)
### Some useful links
- [How to load environments in Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
- [How to install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

## Hierarchy of the files and functions
There is a lot custom functionality built-in to the code. Therefore we have some overview of the different functions that we wrote, what files they originate from, and where they are used. All custom functions that are used in main.py have built-in documentation that you can refer to if things are unclear.

![main hierarchy](https://github.com/DAM-IC/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types/assets/29426481/f131b5d5-3ff0-4154-a909-c008537f56a0)

## The data
Because we work with sensitive patient data, we are unfortunately unable to share our data. However, to guide others to use our code we have supplied some random dummy data such that you can run the code and see the format that your data should be in if you want to run it on your own data. The values in this dummy data are completely random and not taken from actual patients. However, they are sampled from a similar distribution, meaning that the distribution reflects the real data up to some extent. Because the values are random, you should not be able to find meaningful results or correlations.

# References
This code is built for a large part on code from others. Therefore we would like to give credit to the creators of the code that we used, and we ask that you do the same if you end up using this code.

- The DEC functionality is based on the work of [Xie et al., 2015](https://doi.org/10.48550/arXiv.1511.06335) (https://github.com/piiswrong/dec).
- The XVAE functionality is based on the work of [Simidjievski et al., 2019](https://www.frontiersin.org/articles/10.3389/fgene.2019.01205) (https://github.com/CancerAI-CL/IntegrativeVAEs)

## Remaining questions
If there are any remaining questions then you can check the documentation of the functions in python. Alternatively, you can reach us at DAM-IC@outlook.com. 
