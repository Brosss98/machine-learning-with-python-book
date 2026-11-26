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

## Local methods and bias–variance (Chapter 4)

| Symbol | Meaning |
|---|---|
| $N_k(\mathbf{x})$ | The set of the $k$ training points nearest to $\mathbf{x}$ (its $k$ nearest neighbors) |
| $\hat f_k(\mathbf{x})$ | The prediction of the $k$-nearest-neighbor method at $\mathbf{x}$ |
| $f$ | The true regression function, $f(\mathbf{x}) = \mathbb{E}[y\mid\mathbf{x}]$ (the function called $\eta$ in Chapter 3) |
| $\sigma^2$ | Variance of the noise, $y = f(\mathbf{x}) + \varepsilon$ |
| $\text{bias}^2,\ \text{variance}$ | Squared systematic error and sensitivity to the training set of a prediction $\hat f_S(\mathbf{x}_0)$: $\mathbb{E}[(y_0-\hat f_S(\mathbf{x}_0))^2]=\sigma^2+\text{bias}^2+\text{variance}$ |

## Model evaluation (Chapter 5)

| Symbol | Meaning |
|---|---|
| $K,\ \kappa(i)$ | Number of folds of cross-validation; the index of the fold that contains example $i$ |
| $\mathrm{CV}_K$ | The $K$-fold cross-validation estimate of the error: the average loss of each example under the model fitted without its fold |
| $n_{\mathrm{val}},\ m$ | Size of a validation set; number of candidate models or configurations among which the best is selected |
| $\mathrm{TP},\ \mathrm{FP},\ \mathrm{FN},\ \mathrm{TN}$ | Entries of the confusion matrix: true positives, false positives, false negatives, true negatives |
| $\text{TPR}(t),\ \text{FPR}(t)$ | True-positive rate (recall) and false-positive rate of the rule "predict positive if the score is at least $t$" |
| $S^+,\ S^-$ | Score of a randomly chosen positive and of a randomly chosen negative example |
| $\text{AUC},\ \text{AP}$ | Area under the ROC curve; average precision (area under the precision–recall curve) |
| $F_\beta$ | Weighted harmonic mean of precision and recall; $\beta > 1$ gives more weight to recall |
| $\pi$ | Prevalence: the fraction of positive examples in the population |
| $c_{FP},\ c_{FN},\ t^\ast$ | Costs of a false positive and of a false negative; the cost-optimal threshold $t^\ast = c_{FP}/(c_{FP}+c_{FN})$ for calibrated probabilities |
| $h_{ii}$ | Leverage of example $i$ in linear regression (diagonal of the hat matrix); the leave-one-out residual is $e_i/(1-h_{ii})$ |

## Linear models and regularization (Chapter 6)

| Symbol | Meaning |
|---|---|
| $\lambda,\ \hat{\mathbf{w}}_\lambda$ | Strength of the $\ell_2$ penalty and the ridge estimator, which minimizes $\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2+\lambda\lVert\mathbf{w}\rVert^2$ (`alpha` in scikit-learn's `Ridge`); for logistic regression the objective is the mean loss plus $\tfrac\lambda2\lVert\mathbf{w}\rVert^2$. The intercept is never penalized |
| $\sigma_j,\ \mathbf{u}_j,\ \mathbf{v}_j,\ f_j(\lambda)$ | Singular values and left and right singular vectors of the centered feature matrix; shrinkage factors $f_j(\lambda)=\sigma_j^2/(\sigma_j^2+\lambda)$ of ridge. (Do not confuse $\sigma_j$ with $\sigma^2$, the noise variance, nor with $\sigma(\cdot)$, the sigmoid) |
| $\mathbf{H}_\lambda,\ \mathrm{df}(\lambda)$ | Smoother (hat) matrix of ridge, $\hat{\mathbf{y}}=\mathbf{H}_\lambda\mathbf{y}$, and effective degrees of freedom $\mathrm{df}(\lambda)=\operatorname{tr}\mathbf{H}_\lambda=\sum_jf_j(\lambda)$ (plus 1 for the intercept) |
| $\mathrm{VIF}_j$ | Variance inflation factor of feature $j$, $1/(1-R_j^2)$ with $R_j^2$ the $R^2$ of regressing feature $j$ on the others |
| $\sigma(z)$ | Logistic function (sigmoid) $1/(1+e^{-z})$; the log-odds $\log\frac{\eta}{1-\eta}$ are linear in the features |
| $\tilde{\mathbf{X}},\ \boldsymbol\theta$ | Design matrix with a column of ones in front, and the parameter vector $(b,\mathbf{w})$ that includes the intercept |
| $p_i,\ \mathbf{S}$ | Predicted probability of class 1 for example $i$; the diagonal matrix $\mathbf{S}=\mathrm{diag}\big(p_i(1-p_i)\big)$ of the Newton weights. (Elsewhere in the chapter $p$ is the number of parameters) |
| $m_i,\ L,\ J_\lambda$ | Signed score $m_i=(2y_i-1)z_i$; negative log-likelihood (log-loss, cross-entropy) $L=\sum_i\log(1+e^{-m_i})$; penalized objective $J_\lambda$ |
| $C$ | Inverse regularization strength of scikit-learn: $C$ multiplies the sum of the losses, so $\lambda=1/(Cn)$ |
| $K,\ p_k(\mathbf{x}),\ \mathbf{w}_k,\ b_k$ | In softmax regression, $K$ is the number of classes (in Chapter 5 it is the number of folds), the probability of class $k$ and the weights and intercept of its score |
| $\boldsymbol\varphi(\mathbf{x})$ | Feature map: a fixed function from $\mathbb{R}^d$ to $\mathbb{R}^p$ (for example all monomials up to a given degree); models linear in $\boldsymbol\varphi$ are nonlinear in $\mathbf{x}$ |

## Optimization (Chapter 7)

| Symbol | Meaning |
|---|---|
| $f,\ f^\ast,\ \mathbf{w}^\ast$ | The objective to be minimized, its minimum value and a minimizer; the suboptimality is $f(\mathbf{w}_k)-f^\ast$ |
| $\nabla f,\ \nabla^2 f$ | Gradient (the vector of partial derivatives) and Hessian (the matrix of second derivatives) of $f$ |
| $\eta,\ \eta_k$ | Step size (learning rate) of gradient descent, constant or at iteration $k$. (In Chapter 3, $\eta(\mathbf{x})$ is the regression function.) |
| $L$ | Smoothness constant: $\nabla^2 f\preceq L\mathbf{I}$, the largest curvature. (In Chapter 6, $L$ is the log-loss.) $L_{\text{glob}}$ and $L_{\text{loc}}$ are the global bound and the largest curvature at the minimizer |
| $\mu$ | Strong-convexity constant: $\nabla^2 f\succeq\mu\mathbf{I}$, the smallest curvature. (Elsewhere $\mu$ is a mean.) |
| $\kappa=L/\mu$ | Condition number of the problem. (In Chapter 5, $\kappa(i)$ is the fold of example $i$.) |
| $\rho$ | Rate of linear convergence: the error is multiplied by at most $\rho<1$ per iteration |
| $\beta$ | Momentum parameter of the heavy-ball and Nesterov methods. (In Theorem 7.12, $\beta$ and $\gamma$ are the constants of the step $\eta_k=\beta/(\gamma+k)$.) |
| $B,\ \mathbf{g}_k,\ \sigma^2$ | Mini-batch size; stochastic gradient at iteration $k$, unbiased for $\nabla f(\mathbf{w}_k)$; bound on its variance, which for a mini-batch is $\sigma^2/B$ |
| epoch | One pass over the training set: $n/B$ updates, the cost of one full gradient |
| $\mathbf{M},\ \mathbf{D}$ | Preconditioner $\mathbf{M}=\mathbf{D}\mathbf{D}^\top$ and the change of variables $\mathbf{w}=\mathbf{D}\mathbf{v}$ it comes from (Proposition 7.7) |
| $\varphi_j(k)$ | Filter factor of the $j$-th eigendirection after $k$ gradient steps, $1-(1-\eta h_j)^k$, to compare with ridge's $h_j/(h_j+\lambda)$ (Proposition 7.13) |

## Support vector machines (Chapter 8)

| Symbol | Meaning |
|---|---|
| $y_i\in\{-1,+1\}$ | Class labels in this chapter and in Section 3.8.4 (Chapter 6 used $\{0, 1\}$ for the logistic regression and $\{-1, +1\}$ only for the signed score $m_i$) |
| $f(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b$ | Score (decision function), with the intercept $b$ kept apart from $\mathbf{w}$ and not penalized; the prediction is $\operatorname{sign} f(\mathbf{x})$ |
| $m_i=y_if(\mathbf{x}_i)$ | Margin of example $i$ (the signed score of Chapter 6): positive if correct, large if confidently correct. The **geometric margin** of a hyperplane is $\gamma=1/\lVert\mathbf{w}\rVert$ when the closest points have $m_i=1$ |
| $\xi_i$ | Slack variable of example $i$, $\xi_i=\max\{0,1-m_i\}$ at the optimum |
| $\ell_{\text{hinge}}(m)$ | Hinge loss $\max\{0,1-m\}$; $\ell_\varepsilon(r)=\max\{0,\lvert r\rvert-\varepsilon\}$ is the $\varepsilon$-insensitive loss of support vector regression |
| $C,\ \lambda$ | Price of a margin violation, and the penalty $\lambda=1/(Cn)$ of the equivalent regularized hinge-loss problem, as in Section 6.5 |
| $\alpha_i$ | Dual variable (Lagrange multiplier) of the constraint of example $i$; $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$. The **support vectors** are the examples with $\alpha_i>0$; those with $\alpha_i=C$ are *bounded*, those with $0<\alpha_i<C$ *free* |
| $\mathbf{K},\ k(\mathbf{x},\mathbf{z})$ | Gram matrix $K_{ij}=\mathbf{x}_i^\top\mathbf{x}_j$ and its generalization, a kernel $k(\mathbf{x}_i,\mathbf{x}_j)=\boldsymbol\phi(\mathbf{x}_i)^\top\boldsymbol\phi(\mathbf{x}_j)$ (Chapter 9). (In Chapter 5, $K$ is the number of folds; in Chapter 6, the number of classes.) |
| $e_t,\ I_{\text{up}},\ I_{\text{low}}$ | In SMO: $e_t=y_t-\sum_s\alpha_sy_sK_{st}$ and the index sets of the multipliers that can still move up or down (Proposition 8.7) |
| $G$ | Bound on the norm of the subgradients in Theorem 8.6. (Not to be confused with the gradient $G$ of the dual used in the code of SMO.) |

## Kernels (Chapter 9)

| Symbol | Meaning |
|---|---|
| $k(\mathbf{x},\mathbf{z})$ | Kernel: an inner product in a feature space, $k(\mathbf{x},\mathbf{z})=\langle\boldsymbol\phi(\mathbf{x}),\boldsymbol\phi(\mathbf{z})\rangle$; equivalently a symmetric function whose Gram matrices are positive semidefinite (Theorem 9.1) |
| $\mathbf{K}$ | Gram (kernel) matrix of the sample, $K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$; $\mathbf{K}_{nm}$, $\mathbf{K}_{mm}$ are the blocks between the sample and $m$ landmarks, and among the landmarks (Nyström). (In Chapter 5, $K$ is the number of folds; in Chapter 6, the number of classes.) |
| $\gamma$ | Bandwidth parameter of the Gaussian and Laplacian kernels, $\exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2)$; $1/\sqrt\gamma$ is the length scale. In the polynomial kernel $(\gamma\,\mathbf{x}^\top\mathbf{z}+c)^p$, $\gamma$ is a scale and $c$ the offset |
| $\mathcal{H}_k,\ \lVert f\rVert_{\mathcal{H}}$ | Reproducing kernel Hilbert space of $k$ and the norm in it; $f(\mathbf{x})=\langle f,k(\cdot,\mathbf{x})\rangle$ (reproducing property); $\lVert f\rVert^2_{\mathcal{H}}=\boldsymbol\alpha^\top\mathbf{K}\boldsymbol\alpha$ for $f=\sum_i\alpha_ik(\cdot,\mathbf{x}_i)$ |
| $\boldsymbol\alpha$ | Coefficients of the representer theorem, $f=\sum_i\alpha_ik(\cdot,\mathbf{x}_i)$; for the SVM they are $\alpha_iy_i$ with the multipliers $\alpha_i$ of Chapter 8, for kernel ridge regression $(\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}$ |
| $\mathbf{H}_\lambda,\ \mu_j$ | Smoother $\mathbf{K}(\mathbf{K}+\lambda\mathbf{I})^{-1}$ of kernel ridge regression (as in Chapter 6) and the eigenvalues of $\mathbf{K}$, with $\operatorname{df}(\lambda)=\sum_j\mu_j/(\mu_j+\lambda)$ |
| $\mathbf{z}(\mathbf{x}),\ D$ | Random Fourier feature map into $\mathbb{R}^D$ (Theorem 9.7). (Not to be confused with the point $\mathbf{z}$ at which a kernel is evaluated.) |

## Sparsity and feature selection (Chapter 10)

| Symbol | Meaning |
|---|---|
| $\lVert\mathbf{w}\rVert_0,\ s$ | Number of non-zero weights, and the size of a sparse model (Definition 10.1); $\mathbf{w}^\ast$ with $s$ non-zeros is the true weight vector in Sections 10.4 and 10.7 |
| $S,\ S^\ast,\ A$ | A subset of features (a *support*), the true support, and the active set (the non-zero weights) of the lasso; $\mathbf{X}_S$ is the matrix of the columns in $S$. (In Chapter 5, $S^+,S^-$ are scores; in Chapter 6, $\mathbf{S}=\operatorname{diag}(p_i(1-p_i))$; $S_t$ below is a function.) |
| $\mathrm{RSS}$, $C_p$, $\mathrm{BIC}$ | Residual sum of squares of a fit, and the criteria of Section 10.3.2 for choosing the size of a subset: $C_p(S)=\mathrm{RSS}(S)/n+2\hat\sigma^2(\lvert S\rvert+1)/n$ and $\mathrm{BIC}(S)=n\ln(\mathrm{RSS}(S)/n)+(\ln n)(\lvert S\rvert+1)$ |
| $\mathbf{P}_S,\ \mathbf{r},\ \tilde{\mathbf{x}}_j$ | Orthogonal projection on the span of the columns in $S$, the current residual, and the part of the feature $\mathbf{x}_j$ orthogonal to that span (Lemma 10.3) |
| $\mu$ | **Mutual coherence** of unit-norm features, $\max_{i\ne j}\lvert\mathbf{x}_i^\top\mathbf{x}_j\rvert$ (Corollary 10.5). (In Chapter 7, $\mu$ is the strong-convexity constant; in Chapter 9, $\mu_j$ are eigenvalues of the Gram matrix.) |
| $\lambda,\ F_\lambda,\ \lambda_{\max}$ | Lasso penalty in the convention $F_\lambda(\mathbf{w})=\frac1{2n}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2+\lambda\lVert\mathbf{w}\rVert_1$ (the `alpha` of scikit-learn), the objective, and the smallest $\lambda$ for which $\hat{\mathbf{w}}=\mathbf{0}$, $\frac1n\lVert\mathbf{X}^\top\mathbf{y}\rVert_\infty$ (Theorem 10.7). For $\ell_1$ logistic regression $\lambda=1/(nC)$ (Section 10.9) |
| $S_t(a)$ | Soft-thresholding operator, $\operatorname{sign}(a)\max(\lvert a\rvert-t,0)$ (Proposition 10.6); also the proximal operator of $t\lVert\cdot\rVert_1$ (Proposition 10.9) |
| $\lambda_1,\ \lambda_2$ | Weights of the $\ell_1$ and the squared $\ell_2$ penalties of the elastic net, $\text{alpha}\cdot\text{l1\_ratio}$ and $\text{alpha}\,(1-\text{l1\_ratio})$ in scikit-learn (Definition 10.3) |
| $\rho_{ij}$ | Correlation of two standardized features, $\frac1n\mathbf{z}_i^\top\mathbf{z}_j$ (Theorem 10.14) |
| $\kappa_S$ | Condition number of the Hessian of the data term restricted to the support of the solution (Section 10.6.4); explains the linear rate of ISTA after the support is identified |
| $\mathbf{s}_A,\ \boldsymbol\theta,\ \mathbf{u}$ | Signs of the active weights, and the vectors of Exercise 10.5, with $\hat{\mathbf{w}}_A(\lambda)=\boldsymbol\theta-\lambda\mathbf{u}$ on a segment of the path |

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
