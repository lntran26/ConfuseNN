## Interpreting supervised machine learning inferences in population genomics using haplotype matrix permutations
Supervised machine learning methods, such as convolutional neural networks (CNNs), that use haplotype matrices as input data have become powerful tools for population genomics inference.
However, these methods often lack interpretability, making it difficult to understand which population genetic features drive their predictions—a critical limitation for method development and biological interpretation.
Here we introduce a systematic permutation approach that progressively disrupts population genetics features within input test haplotype matrices, including linkage disequilibrium, haplotype structure, and allele frequencies.
By measuring performance degradation after each permutation, the importance of each feature can be assessed.
We applied our approach to three published CNNs for positive selection and demographic history inference.

### Preprint
[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.24.644668v1)

### Reproduce

To reproduce the result for each of the three CNNs evaluated in this work, 
refer to the three respective subdirs, each with their own specifications.

### Example code for haplotype matrix permutations

For ease of adoptability, we have provided the code to perform all permutations described in our paper in `minimal_example.ipynb`, with visualization.
How these permutations are applied in practice likely varies depending on the simulation and training procedure.
For examples of how we customized our permutation approach to each CNN, refer to corresponding subdir with further descriptions.

### Extra: Conference presentations of this work
* [TAGC24 Poster](https://github.com/lntran26/lntran26.github.io/blob/4e461eaf627614b75ec47d9a8f72fd5491880fb9/files/TAGC_24_Tran_final.pdf)
* [ISMB24 MLCSB COSI Talk](https://github.com/lntran26/lntran26.github.io/blob/1a5497584df962c9d5643b013523ecaf0fa8c2ef/files/ISMB24.pdf)
* [EVO-WIBO Poster](https://github.com/lntran26/lntran26.github.io/blob/master/files/Tran_ConfuseNN_final.pdf)
