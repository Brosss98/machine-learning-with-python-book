# Notation and Conventions

This page collects the notation used throughout the book. Do not try to memorize it: each symbol is introduced again where it first appears, and you can return here whenever you need.

## Numbers, vectors, and matrices

| Symbol | Meaning |
|---|---|
| $a,\ x,\ \lambda$ | Scalars (italic lowercase letters) |
| $\mathbf{x},\ \mathbf{w}$ | Column vectors (bold lowercase letters) |
| $\mathbf{A},\ \mathbf{X}$ | Matrices (bold uppercase letters) |
| $\mathbf{A}^\top,\ \mathbf{A}^{-1}$ | Transpose and inverse of a matrix |
| $\mathbf{I}$ | Identity matrix |
| $\mathbb{R},\ \mathbb{R}^d,\ \mathbb{R}^{n\times d}$ | Real numbers, real vectors of length $d$, real matrices with $n$ rows and $d$ columns |
| $\mathbf{x}^\top\mathbf{y}$ | Dot product of two vectors |
| $\lVert\mathbf{x}\rVert_2,\ \lVert\mathbf{x}\rVert_1$ | Euclidean norm and $\ell_1$ norm |
| $\lVert\mathbf{A}\rVert_F$ | Frobenius norm of a matrix |

## Data

| Symbol | Meaning |
|---|---|
| $n$ | Number of samples (examples) |
| $d$ | Number of features (input dimension) |
| $K$ | Number of classes or clusters |
| $\mathbf{x}_i\in\mathbb{R}^d$ | The $i$-th input vector |
| $y_i$ | The label or target of the $i$-th example |
| $\mathbf{X}\in\mathbb{R}^{n\times d}$ | Data matrix: its $i$-th row is $\mathbf{x}_i^\top$ |
| $\mathbf{y}\in\mathbb{R}^{n}$ | Vector of all targets |
| $S=\{(\mathbf{x}_i,y_i)\}_{i=1}^{n}$ | Training set |

In code, the data matrix is an array `X` of shape `(n, d)`, with one sample per row, which is the convention of scikit-learn.

## Learning

| Symbol | Meaning |
|---|---|
| $f$ | A prediction function; $\hat{y}=f(\mathbf{x})$ is the prediction for $\mathbf{x}$ |
| $\mathcal{H}$ | Hypothesis space: the set of functions a method can choose from |
| $\mathbf{w},\ b$ | Weights and bias of a model |
| $\boldsymbol{\theta}$ | Generic vector of parameters |
| $\ell(y,\hat{y})$ | Loss on a single example |
| $R(f)$ | Expected risk of $f$ |
| $\hat{R}_S(f)$ | Empirical risk of $f$ on the training set $S$ |
| $\lambda$ | Regularization strength |
| $\eta$ | Learning rate |
| $\nabla_{\mathbf{w}} L$ | Gradient of the function $L$ with respect to $\mathbf{w}$ |
| $\operatorname{argmin}_{\mathbf{w}}\, L(\mathbf{w})$ | The value of $\mathbf{w}$ that minimizes $L$ |

## Statistical learning theory (Chapter 3)

| Symbol | Meaning |
|---|---|
| $f^\ast,\ R^\ast$ | Bayes predictor and Bayes risk: the smallest risk of any function |
| $\eta(\mathbf{x})$ | Regression function $\mathbb{E}[y \mid \mathbf{x}]$ (for $y\in\{0,1\}$ it is $\mathbb{P}(y=1\mid\mathbf{x})$); not the learning rate |
| $\hat f_S$ | The function chosen by empirical risk minimization on the training set $S$ |
| $f_{\mathcal{H}}$ | The best function in $\mathcal{H}$, that is, the one with the smallest expected risk |
| $\varepsilon,\ \delta$ | Accuracy and failure probability of a guarantee: "with probability at least $1-\delta$, the error is at most $\varepsilon$" |
| $\Pi_{\mathcal{H}}(n)$ | Growth function: the largest number of labelings of $n$ points that $\mathcal{H}$ can produce |
| $d_{\mathrm{VC}}$ | Vapnik–Chervonenkis dimension: the size of the largest set of points that $\mathcal{H}$ can label in all possible ways |
| $\hat{\mathfrak{R}}_S(\mathcal{F}),\ \mathfrak{R}_n(\mathcal{F})$ | Empirical and expected Rademacher complexity of a class $\mathcal{F}$ |

## Probability

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | The unknown distribution that generates the data |
| $\mathbb{P}(A)$ | Probability of the event $A$ |
| $\mathbb{E}[X],\ \operatorname{Var}(X)$ | Expectation and variance of a random variable |
| $\mathcal{N}(\mu,\sigma^2)$ | Gaussian distribution with mean $\mu$ and variance $\sigma^2$ |
| $\mathbb{I}[\cdot]$ | Indicator function: equal to 1 if the condition holds and 0 otherwise |

## Conventions to keep in mind

- **Vectors are columns.** A vector $\mathbf{x}\in\mathbb{R}^d$ is a $d\times 1$ column, and $\mathbf{x}^\top$ is a row.
- **Indices start at 1 in the mathematics and at 0 in Python.** The first sample is $\mathbf{x}_1$ in the formulas and `X[0]` in the code. This is a frequent source of off-by-one mistakes.
- **Code is written in `monospace`.** Names such as `X`, `y`, and `w` match the symbols in the text whenever possible.
- **Local notation.** A chapter may introduce extra symbols for its own purposes. They are defined where they are first used.
