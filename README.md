# Pytorch-GANBERT
A Pytorch implementation of [GAN-BERT](https://www.aclweb.org/anthology/2020.acl-main.191.pdf) paper

"bert.py" and "qc-fine_bert.py" files are basically the same with different datasets; this is just my laziness.
"labeled_and_unlabeled.tsv" is the mixed version of labeled and unlabeled data from original [tensorflow implementation](https://github.com/crux82/ganbert) of the paper.

### Important Note
Running this model only once with only one seed might not be enough. In my experiments, out of 10 runs, only 2 yielded reasonable results. I believe this is due to the difference of scheduling of the optimizers between this implementation and the original TensorFlow implementation.