## ConfuseNN: Interpreting convolutional neural networks inferences in population genomics by data shuffling

Convolutional neural network (CNN) is an increasingly popular supervised machine learning approach that has been applied to many inference tasks in population genetics. 
Under this framework, population genomic variation data are typically represented as 2D images with sampled haplotypes as rows and segregating sites as columns. 
While many published studies reported promising performance of CNNs on various inference tasks, 
understanding which features in the data were picked up by the CNNs and meaningfully contributed to the reported performance remains challenging. 
Here we propose a novel approach to interpreting CNN performance motivated by population genetic theory on genomic data. 
Specifically, we designed a suite of scramble tests where each test deliberately disrupts a feature in the genomic image data 
(e.g. allele frequency, linkage disequilibrium, etc.) to assess how each feature affects the CNN performance. 
We apply these tests to three networks designed to infer demographic history and natural selection, 
identifying the fundamental population genomic features that drive inference for each network.

### Early result reference
* [TAGC24 Poster](https://github.com/lntran26/lntran26.github.io/blob/4e461eaf627614b75ec47d9a8f72fd5491880fb9/files/TAGC_24_Tran_final.pdf)