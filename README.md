# Genetic Programming with Rademacher Complexity

This repository contains code for reproducing the experiments in the paper "[*Genetic Programming with Rademacher Complexity for Symbolic Regression*](https://ieeexplore.ieee.org/document/8790341)" by Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang.

## Contents

Implementation of Genetic Programming for Symbolic Regression (GP-SR) and the newly proposed Genetic Programming with Rademacher Complexity (GPRC):

- [Genetic Programming for Symbolic Regression (GP-SR)](https://github.com/Decadz/Genetic-Programming-for-Symbolic-Regression/blob/master/algorithms/genetic_programming_classic.py) A prototypical implementation of Genetic Programming for Symbolic Regression. This program aims to map the input data to the output data through the use of a symbolic representation (expression trees) and evolutionary techniques.

- [Genetic Programming with Rademacher Complexity (GP-RC)](https://github.com/Decadz/Genetic-Programming-for-Symbolic-Regression/blob/master/algorithms/genetic_programming_rademacher_complexity.py) An implementation of Genetic Programming for Symbolic Regression that uses the Rademacher Complexity to estimate the complexity of the hypotheses generated in the evolutionary process.

### Code Reproducibility: 

The code has not been comprehensively checked and re-run since refactoring. If you're having any issues, find
a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Genetic-Programming-with-Rademacher-Complexity/issues)
or email me.

## Reference

If you use our library or find our research of value please consider citing our papers with the following Bibtex entry:

```
@inproceedings{raymond2019genetic,
  title={Genetic Programming with Rademacher Complexity for Symbolic Regression},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  booktitle={2019 IEEE Congress on Evolutionary Computation (CEC)},
  pages={2657--2664},
  year={2019},
  organization={IEEE}
}
```
