# %% [markdown]
# # Appendix C — Solutions to the Exercises
#
# This appendix collects the solutions of the exercises, chapter by chapter. It grows as the book grows: each new chapter adds its own section.
#
# **How to use it.** Try the exercise first, then read the hint in the chapter, and only then look here. For the manual exercises the solution is written out in full. For the computational exercises the solution is a complete, executable piece of code, followed by a short discussion of what the output shows; there are many correct ways to write such code, and yours may differ.
#
# **A note on the code.** Each solution lives in a function (`solution_1_6()`, and so on), so that the cells do not overwrite each other's variables. The function names match the exercise numbers.

# %%
import platform
from importlib.metadata import version

print("Python", platform.python_version())
for pkg in ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn"]:
    print(f"{pkg:<14}{version(pkg)}")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42

# Same plotting style and color-blind-safe palette as in the chapters
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERMILION, SKY, PURPLE = "#D55E00", "#56B4E9", "#CC79A7"
plt.rcParams.update({
    "figure.dpi": 110,
    "figure.figsize": (6.4, 4.0),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# %% [markdown]
# ## C.1 Chapter 1
#
# ### Manual exercises
#
# **Solution 1.1 — Name the problem.**
#
# | Task | Paradigm | Type |
# |---|---|---|
# | (a) tomorrow's electricity consumption | supervised | regression (the target is a real number) |
# | (b) grouping customers without predefined groups | unsupervised | clustering (Chapter 12) |
# | (c) spam or not spam | supervised | classification (two classes) |
# | (d) learning a game by self-play, from wins and losses | reinforcement learning | the wins and losses are the reward signal (Chapter 22) |
# | (e) reducing 1000 measurements to 10 numbers | unsupervised | dimensionality reduction (Chapter 13) |
#
# **Solution 1.2 — Shapes.**
# (a) $\mathbf{X} \in \mathbb{R}^{200 \times 5}$ and $\mathbf{y} \in \mathbb{R}^{200}$.
# (b) The intercept column adds one column: $200 \times 6$.
# (c) $\mathbf{X}^\top\mathbf{X}$ is $d \times d$, that is $5 \times 5$ (and $6 \times 6$ for the design matrix with the intercept). Its size depends on the number of features, not on the number of samples.
# (d) Each entry of $\mathbf{X}\mathbf{w}$ is the dot product of a row of $\mathbf{X}$ with $\mathbf{w}$:
#
# $$
# \mathbf{X}\mathbf{w} = \begin{pmatrix} 1\cdot 1 + 2\cdot(-1) \\ 3\cdot 1 + 4\cdot(-1) \\ 5\cdot 1 + 6\cdot(-1) \end{pmatrix} = \begin{pmatrix} -1 \\ -1 \\ -1 \end{pmatrix}.
# $$
#
# The $i$-th entry is the prediction of the linear model $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$ for the $i$-th sample; here, the first feature minus the second.
#
# **Solution 1.3 — Least squares by hand.** The points are $(0,2)$, $(1,2)$, $(2,5)$.
#
# (a) $\mathbf{X} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$, $\mathbf{y} = \begin{pmatrix} 2 \\ 2 \\ 5 \end{pmatrix}$. Then $\mathbf{X}^\top\mathbf{X} = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}$ and $\mathbf{X}^\top\mathbf{y} = \begin{pmatrix} 9 \\ 12 \end{pmatrix}$. With $\det = 6$,
#
# $$
# \hat{\mathbf{w}} = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix}\begin{pmatrix} 9 \\ 12 \end{pmatrix}
# = \frac{1}{6}\begin{pmatrix} 9 \\ 9 \end{pmatrix} = \begin{pmatrix} 3/2 \\ 3/2 \end{pmatrix},
# $$
#
# so the line is $\hat{y} = \tfrac32 + \tfrac32 x$.
#
# (b) $\hat{\mathbf{y}} = (1.5,\ 3,\ 4.5)^\top$, hence $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}} = (0.5,\ -1,\ 0.5)^\top$. Then $\sum_i r_i = 0$ and $\mathbf{r}\cdot(0,1,2) = 0 - 1 + 1 = 0$: $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$.
#
# (c) $\text{MSE} = \frac{0.25 + 1 + 0.25}{3} = 0.5$. The mean of $\mathbf{y}$ is $3$ and $\sum_i (y_i - \bar y)^2 = 1 + 1 + 4 = 6$, so $R^2 = 1 - \frac{1.5}{6} = 0.75$.
#
# **Solution 1.4 — Derive it yourself.**
# (a) $L(w) = \sum_i (y_i - w x_i)^2$, so $L'(w) = -2\sum_i x_i (y_i - w x_i)$. Setting $L'(w) = 0$ gives $\sum_i x_i y_i = w \sum_i x_i^2$, that is
#
# $$
# \hat{w} = \frac{\sum_i x_i y_i}{\sum_i x_i^2}.
# $$
#
# (b) With a single column $\mathbf{x} = (x_1, \dots, x_n)^\top$ as the data matrix, $\mathbf{X}^\top\mathbf{X} = \mathbf{x}^\top\mathbf{x} = \sum_i x_i^2$ (a $1\times 1$ matrix) and $\mathbf{X}^\top\mathbf{y} = \sum_i x_i y_i$, so $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ is the same ratio.
#
# (c) $L''(w) = 2\sum_i x_i^2$, which is strictly positive unless every $x_i$ is zero. Hence $L$ is strictly convex and $\hat{w}$ is its unique minimizer.
#
# **Solution 1.5 — Which loss, which constant?**
# (a) $f(c) = \sum_i (y_i - c)^2$ has $f'(c) = -2\sum_i (y_i - c)$, which vanishes at $c = \frac1n\sum_i y_i$; and $f'' = 2n > 0$. The minimizer is the mean.
#
# (b) Let $g(c) = \sum_i \lvert y_i - c\rvert$, with $n = 2m - 1$ and $m = (n+1)/2$. The function $g$ is piecewise linear. At a point $c$ that is not a data point, $g'(c) = \#\{y_i < c\} - \#\{y_i > c\}$ (each term with $y_i < c$ contributes $+1$, each term with $y_i > c$ contributes $-1$).
# - For $c$ between $y_{(m-1)}$ and $y_{(m)}$: $m - 1$ points lie below and $m$ above, so $g' = -1 < 0$.
# - For $c$ between $y_{(m)}$ and $y_{(m+1)}$: $m$ points lie below and $m - 1$ above, so $g' = +1 > 0$.
#
# More generally the slope is $2k - n$ when $k$ points lie below $c$, which is negative for $k < m$ and positive for $k \ge m$. So $g$ decreases up to $y_{(m)}$ and increases afterwards: the minimizer is $y_{(m)}$, the median.
#
# (c) Replace $y_n$ by $10^6$. The order of the points is unchanged (it was already the largest, and it only grows), so the **median is unchanged**. The **mean** instead moves by $(10^6 - y_n)/n$, which can be enormous: one outlier drags the optimal constant of the squared loss arbitrarily far. The squared loss is therefore sensitive to outliers (it squares large errors), while the absolute loss is robust. When the data may contain outliers, the choice of loss matters as much as the choice of model.

# %% [markdown]
# ### Python exercises
#
# **Solution 1.6 — Meet a new data set.**

# %%
def solution_1_6():
    from sklearn.datasets import load_wine

    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    print("shape of X:", X.shape)
    print("classes:", dict(zip(wine.target_names.tolist(), np.bincount(y).tolist())))
    print("first five features:", X.columns[:5].tolist())

    stats = X.describe().loc[["mean", "std"]].T
    print("\nmean and standard deviation of each feature:")
    print(stats.round(3).to_string())
    print("\nfeature with the largest standard deviation:", stats["std"].idxmax())

solution_1_6()

# %% [markdown]
# The Wine data set has 178 samples, 13 features, and 3 classes. The feature `proline` lives on a scale of hundreds (its mean is in the 700s and its standard deviation above 300), while most other features are of order 1 to 100 and some are below 1. In a distance-based method such as k-NN, `proline` would dominate the Euclidean distance and the other features would barely matter: **standardize the features** first (Section 1.8).
#
# **Solution 1.7 — A line on Iris.**

# %%
def solution_1_7():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LinearRegression

    iris = load_iris(as_frame=True)
    x = iris.data["petal length (cm)"].to_numpy()
    t = iris.data["petal width (cm)"].to_numpy()
    A = np.column_stack([np.ones_like(x), x])

    # (a) three ways to the same answer
    w_normal = np.linalg.solve(A.T @ A, A.T @ t)
    w_lstsq = np.linalg.lstsq(A, t, rcond=None)[0]
    lr = LinearRegression().fit(x.reshape(-1, 1), t)
    w_sk = np.array([lr.intercept_, lr.coef_[0]])
    print("normal equations:", w_normal.round(5))
    print("lstsq           :", w_lstsq.round(5))
    print("scikit-learn    :", w_sk.round(5))
    print("all agree:", np.allclose(w_normal, w_lstsq) and np.allclose(w_normal, w_sk))

    # (b) error, baseline, R^2
    t_hat = A @ w_normal
    mse = np.mean((t - t_hat) ** 2)
    mse_base = np.mean((t - t.mean()) ** 2)
    r2 = 1 - np.sum((t - t_hat) ** 2) / np.sum((t - t.mean()) ** 2)
    print(f"\nMSE = {mse:.4f} | baseline MSE = {mse_base:.4f} | R^2 = {r2:.4f}")

    # (c) plot
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    species = iris.target.to_numpy()
    for k, (name, color, marker) in enumerate(zip(iris.target_names, [BLUE, ORANGE, GREEN], ["o", "s", "^"])):
        ax.scatter(x[species == k], t[species == k], s=26, color=color, marker=marker,
                   edgecolor="white", linewidth=0.4, label=name)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, w_normal[0] + w_normal[1] * xs, color="black", linewidth=2, label="least-squares line")
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.set_title("Iris: petal width as a linear function of petal length")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()

solution_1_7()

# %% [markdown]
# The three methods agree to numerical precision. The line explains more than 90% of the variance of the petal width (compare the MSE with the baseline MSE, and read the $R^2$): on this data set the two petal measurements are strongly linearly related. Note that the cloud is not homogeneous across species, and the line is a compromise between the three groups.
#
# **Solution 1.8 — Does scaling matter?**

# %%
def solution_1_8():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    ks = np.arange(1, 16)

    def test_accuracies(seed):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        raw = [KNeighborsClassifier(k).fit(X_tr, y_tr).score(X_te, y_te) for k in ks]
        scaled = [make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_tr, y_tr).score(X_te, y_te)
                  for k in ks]
        return raw, scaled

    # the split of Section 1.8 (random_state = SEED)
    raw_1, scaled_1 = test_accuracies(SEED)
    # ten different splits
    seeds = range(10)
    all_raw, all_scaled = map(np.array, zip(*(test_accuracies(s) for s in seeds)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    axes[0].plot(ks, raw_1, "o--", color=VERMILION, label="raw features")
    axes[0].plot(ks, scaled_1, "s-", color=BLUE, label="standardized")
    axes[0].set_title(f"One split (random_state={SEED})")
    axes[1].plot(ks, all_raw.mean(axis=0), "o--", color=VERMILION, label="raw features")
    axes[1].plot(ks, all_scaled.mean(axis=0), "s-", color=BLUE, label="standardized")
    axes[1].fill_between(ks, all_scaled.mean(0) - all_scaled.std(0), all_scaled.mean(0) + all_scaled.std(0),
                         color=BLUE, alpha=0.15)
    axes[1].fill_between(ks, all_raw.mean(0) - all_raw.std(0), all_raw.mean(0) + all_raw.std(0),
                         color=VERMILION, alpha=0.15)
    axes[1].set_title("Mean over 10 splits (band: ± 1 std across splits)")
    for ax in axes:
        ax.set_xlabel("number of neighbors k")
        ax.legend()
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout()
    plt.show()

    diff = all_scaled - all_raw                      # shape (10 splits, 15 values of k)
    print(f"mean accuracy gain from scaling: {diff.mean():.3f}")
    print(f"fraction of (split, k) pairs where scaling is at least as good: {(diff >= 0).mean():.3f}")
    print(f"fraction where scaling is strictly better: {(diff > 0).mean():.3f}")

solution_1_8()

# %% [markdown]
# The conclusion is stable. Averaged over the ten splits, standardized features give a higher test accuracy for essentially every $k$ (the gain is about three percentage points in our run), and the printed fractions show that scaling was never worse and almost always strictly better across all the (split, $k$) pairs. A single split would have been enough to see the effect here, but *only* repeating the experiment lets us say that the conclusion does not depend on the lucky choice of one split. This habit, repeating the evaluation on several splits, is formalized as cross-validation in Chapter 5.
#
# **Solution 1.9 — Your own overfitting experiment.**

# %%
def solution_1_9():
    from numpy.polynomial import polynomial as P
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    d = load_diabetes(as_frame=True, scaled=False)
    target = d.target.to_numpy()
    MAX_DEG, N_SPLITS = 8, 100

    def median_test_errors(z, n_train):
        errs = np.zeros((N_SPLITS, MAX_DEG + 1))
        for s in range(N_SPLITS):
            z_tr, z_te, y_tr, y_te = train_test_split(z, target, train_size=n_train, random_state=s)
            for degree in range(MAX_DEG + 1):
                coef = P.polyfit(z_tr, y_tr, degree)
                errs[s, degree] = np.mean((y_te - P.polyval(z_te, coef)) ** 2)
        return np.median(errs, axis=0)

    rows = []
    for feature in ["bp", "s5"]:
        f = d.data[feature].to_numpy()
        z = (f - f.mean()) / f.std()
        for n_train in [30, 100, 300]:
            med = median_test_errors(z, n_train)
            rows.append({"feature": feature, "training size": n_train,
                         "best degree": int(med.argmin()),
                         "test MSE at best degree": round(float(med.min()), 0),
                         "test MSE at degree 1 (a line)": round(float(med[1]), 0),
                         "test MSE at degree 0 (baseline)": round(float(med[0]), 0)})
    print(pd.DataFrame(rows).to_string(index=False))

solution_1_9()

# %% [markdown]
# In the table, the **best degree does not decrease as the training set grows**, and it tends to increase: with 30 patients a straight line is the best choice for both features, while with 300 patients a slightly more flexible curve is preferred. The reason is that the variance of a flexible model comes from having too few samples to pin down its many coefficients; as $n$ grows this variance shrinks, and the best compromise moves towards more flexibility. Note also that the gain of the best degree over a straight line is small (a few percent at most, and nil for `bp` with 100 or fewer patients): the relationship between a single feature and the target is weak and almost linear, so there is little to gain from flexibility, and the errors for neighboring degrees are close. Every model, however, is clearly better than the baseline (degree 0). The idea that **the right complexity depends on the amount of data** returns in Chapters 3 and 4.
#
# **Solution 1.10 — When $\mathbf{X}^\top\mathbf{X}$ is singular.**

# %%
def solution_1_10():
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True, scaled=False)
    bmi, target = d.data["bmi"].to_numpy(), d.target.to_numpy()
    A1 = np.column_stack([np.ones_like(bmi), bmi])            # intercept + bmi
    A2 = np.column_stack([np.ones_like(bmi), bmi, bmi])       # intercept + bmi + a duplicate of bmi

    # (a) rank and conditioning
    print("rank of A1:", np.linalg.matrix_rank(A1), "| rank of A2:", np.linalg.matrix_rank(A2),
          "(A2 has 3 columns but only 2 independent ones)")
    print(f"condition number of A1^T A1: {np.linalg.cond(A1.T @ A1):.3g}")
    print(f"condition number of A2^T A2: {np.linalg.cond(A2.T @ A2):.3g}")

    # (b) the normal equations
    try:
        w_solve = np.linalg.solve(A2.T @ A2, A2.T @ target)
        print("np.linalg.solve returned (numerically meaningless):", w_solve)
    except np.linalg.LinAlgError as err:
        print("np.linalg.solve raised LinAlgError:", err)

    # (c) lstsq
    w2 = np.linalg.lstsq(A2, target, rcond=None)[0]
    w1 = np.linalg.lstsq(A1, target, rcond=None)[0]
    print("\nlstsq on A2:", w2.round(4), "| norm:", round(float(np.linalg.norm(w2)), 4))
    print("lstsq on A1:", w1.round(4))
    print("same predictions with and without the duplicate column:", np.allclose(A2 @ w2, A1 @ w1))

    # (d) other solutions with the same predictions
    candidates = {"(w1, 0)": np.array([w1[0], w1[1], 0.0]),
                  "(w1/2, w1/2)  [what lstsq picks]": np.array([w1[0], w1[1] / 2, w1[1] / 2]),
                  "(3 w1/4, w1/4)": np.array([w1[0], 0.75 * w1[1], 0.25 * w1[1]])}
    print("\nweights giving exactly the same predictions, and their norms:")
    for name, w in candidates.items():
        print(f"  {name:<36} predictions equal: {np.allclose(A2 @ w, A1 @ w1)} | norm = {np.linalg.norm(w):.4f}")

solution_1_10()

# %% [markdown]
# (a) The matrix $\mathbf{A}_2$ has three columns but rank 2, because the third column equals the second. Consequently $\mathbf{A}_2^\top\mathbf{A}_2$ is singular, and its condition number is astronomically large (of the order of the inverse of machine precision, or beyond), while that of $\mathbf{A}_1^\top\mathbf{A}_1$ is moderate.
#
# (b) The normal equations no longer have a unique solution. Depending on rounding errors, `np.linalg.solve` either raises a `LinAlgError` ("Singular matrix") or returns meaningless numbers; you must not rely on either behavior.
#
# (c) `np.linalg.lstsq` works through the SVD and returns the solution of **minimum norm** among all least-squares solutions: it splits the coefficient of `bmi` equally between the two identical columns. The predictions are exactly those of the model that uses `bmi` once.
#
# (d) In terms of the column space: adding a column that is already a linear combination of the others does not enlarge the column space, so the projection $\hat{\mathbf{y}}$ is the same. What changes is the *parametrization*: infinitely many weight vectors $(w_0, a, b)$ with $a + b = w_1$ produce the same $\hat{\mathbf{y}}$, and the normal equations are underdetermined. Ranking them by their norm, as `lstsq` does, picks one; **regularization** (Chapter 6) will do something similar in a principled way and will also make the problem well-conditioned.

# %% [markdown]
# ## C.2 Chapter 2
#
# ### Manual exercises
#
# **Solution 2.1 — Shapes and broadcasting.**
#
# (a)
#
# | Expression | Shapes | Result |
# |---|---|---|
# | `A + v` | $(5,3) + (3,)$ | valid, shape $(5,3)$: `v` is added to every row |
# | `A + c` | $(5,3) + (5,1)$ | valid, shape $(5,3)$: `c` is added to every column |
# | `v + c` | $(3,) + (5,1)$ | valid, shape $(5,3)$: a table of all sums $v_j + c_i$ |
# | `A @ v` | $(5,3)\,@\,(3,)$ | valid, shape $(5,)$ |
# | `A @ c` | $(5,3)\,@\,(5,1)$ | **error**: the inner dimensions are 3 and 5 |
# | `c @ r` | $(5,1)\,@\,(1,3)$ | valid, shape $(5,3)$: the outer product |
# | `r @ A` | $(1,3)\,@\,(5,3)$ | **error**: the inner dimensions are 3 and 5 |
#
# (b) `np.mean((y - X @ w) ** 2)`. The shapes are: `X @ w` is $(n, d)\,@\,(d,) \to (n,)$; `y - X @ w` is $(n,)$; the square is $(n,)$; and the mean is a scalar, shape $()$.
#
# (c) `y - y_hat` has shape $(n, n)$: the vector `y` is treated as a row $(1, n)$ and `y_hat` as a column, so the entry $(i, j)$ is $y_j - \hat y_i$. The mean of the squares is the average of $(y_j - \hat y_i)^2$ over **all pairs**. Expanding the square and averaging over $i$ and $j$ gives
# $$
# \frac{1}{n^2}\sum_{i,j}(y_j - \hat y_i)^2 = \overline{y^2} - 2\,\bar y\,\bar{\hat y} + \overline{\hat y^2} = \operatorname{Var}(y) + \operatorname{Var}(\hat y) + (\bar y - \bar{\hat y})^2,
# $$
# with the variances computed with the divisor $n$. It measures how different the *distributions* of the targets and of the predictions are, not how good each individual prediction is. On the data of the Pitfall, $\bar y = 2.5$, $\bar{\hat y} = 2.75$, $\operatorname{Var}(y) = 1.25$ and $\operatorname{Var}(\hat y) = 1.1875$, so the result is $1.25 + 1.1875 + 0.0625 = 2.5$, which is the "wrong MSE" printed in Section 2.3.6, while the correct MSE is 0.25.
#
# **Solution 2.2 — Counting heads.**
#
# (a) $\mathbb{P}(X = k) = \binom{3}{k}/8$, that is $\tfrac18, \tfrac38, \tfrac38, \tfrac18$ for $k = 0, 1, 2, 3$.
#
# (b) $\mathbb{E}[X] = 0\cdot\tfrac18 + 1\cdot\tfrac38 + 2\cdot\tfrac38 + 3\cdot\tfrac18 = \tfrac{12}{8} = 1.5 = np$. Moreover $\mathbb{E}[X^2] = 0 + \tfrac38 + \tfrac{12}{8} + \tfrac98 = 3$, so $\operatorname{Var}(X) = 3 - 1.5^2 = 0.75 = np(1-p) = 3 \cdot \tfrac14$.
#
# (c) $\mathbb{E}[Z] = 2 \cdot 1.5 - 3 = 0$ and $\operatorname{Var}(Z) = 2^2 \cdot 0.75 = 3$. Since $3 - X$ is the number of tails, $Z = X - (3 - X)$ is **heads minus tails**: the position, after three steps, of a walker that moves $\pm 1$ with equal probability. It is symmetric around 0.
#
# **Solution 2.3 — A spam filter by Bayes.**
#
# (a) With $S$ = spam, $\mathbb{P}(S) = 0.4$, $\mathbb{P}(\text{free}\mid S) = 0.6$, $\mathbb{P}(\text{free}\mid \bar S) = 0.05$:
# $$
# \mathbb{P}(S \mid \text{free}) = \frac{0.6 \cdot 0.4}{0.6 \cdot 0.4 + 0.05 \cdot 0.6} = \frac{0.24}{0.27} = \frac{8}{9} \approx 0.889.
# $$
#
# (b) Assuming that the two words are independent *within each class*,
# $$
# \mathbb{P}(S \mid \text{free}, \text{winner}) = \frac{0.4 \cdot 0.6 \cdot 0.3}{0.4 \cdot 0.6 \cdot 0.3 + 0.6 \cdot 0.05 \cdot 0.02} = \frac{0.072}{0.0726} \approx 0.992.
# $$
#
# (c) The assumption is **conditional independence given the class**: $\mathbb{P}(\text{free}, \text{winner}\mid S) = \mathbb{P}(\text{free}\mid S)\,\mathbb{P}(\text{winner}\mid S)$, and the same for ham. If the words are correlated inside a class (and in spam they probably are), the second word carries less new information than the product suggests, so the evidence is counted twice and the posterior tends to be *overconfident*. This is the "naive" assumption of the naive Bayes classifier: it is often wrong, yet the resulting classifier can still work well, because only the ranking of the classes matters.
#
# **Solution 2.4 — Maximum likelihood for the exponential distribution.**
#
# (a) $\ell(\lambda) = \sum_{i=1}^n \log(\lambda e^{-\lambda x_i}) = n\log\lambda - \lambda\sum_i x_i$. Then $\ell'(\lambda) = n/\lambda - \sum_i x_i = 0$ gives
# $$
# \hat\lambda = \frac{n}{\sum_i x_i} = \frac{1}{\bar x}.
# $$
#
# (b) $\ell''(\lambda) = -n/\lambda^2 < 0$ for every $\lambda > 0$: the log-likelihood is concave, so $\hat\lambda$ is the maximum.
#
# (c) $\sum_i x_i = 5$ and $n = 4$, so $\bar x = 1.25$ and $\hat\lambda = 0.8$. The mean of the fitted distribution is $1/\hat\lambda = 1.25$, which is exactly the sample mean: for the exponential family, maximum likelihood matches the model's mean with the data's mean.
#
# **Solution 2.5 — Covariance matrices and whitening.**
#
# (a) $\det(\boldsymbol\Sigma - \lambda\mathbf{I}) = (5 - \lambda)(2 - \lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)$, so $\lambda_1 = 6$ and $\lambda_2 = 1$. For $\lambda_1 = 6$: $(5 - 6)v_1 + 2v_2 = 0$, so $v_2 = v_1/2$ and $\mathbf{q}_1 = \tfrac{1}{\sqrt5}(2, 1)^\top$. For $\lambda_2 = 1$: $4v_1 + 2v_2 = 0$, so $\mathbf{q}_2 = \tfrac{1}{\sqrt5}(1, -2)^\top$ (eigenvectors are defined up to a sign). The variance of $\mathbf{x}$ is 6 along $\mathbf{q}_1$ (standard deviation $\sqrt6 \approx 2.45$) and 1 along $\mathbf{q}_2$.
#
# (b) Using $\mathbb{E}[\mathbf{x}] = \mathbf{0}$ and the linearity of expectation, $\operatorname{Cov}(\mathbf{A}\mathbf{x}) = \mathbb{E}[\mathbf{A}\mathbf{x}\mathbf{x}^\top\mathbf{A}^\top] = \mathbf{A}\,\mathbb{E}[\mathbf{x}\mathbf{x}^\top]\,\mathbf{A}^\top = \mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top$. (With a nonzero mean, apply the same steps to $\mathbf{x} - \boldsymbol\mu$.) For a vector $\mathbf{v}$, take $\mathbf{A} = \mathbf{v}^\top$: $\mathbf{v}^\top\boldsymbol\Sigma\mathbf{v} = \operatorname{Var}(\mathbf{v}^\top\mathbf{x}) \ge 0$, because a variance is an expectation of a square. Hence $\boldsymbol\Sigma$ is positive semidefinite.
#
# (c) With $\mathbf{Q} = (\mathbf{q}_1\ \mathbf{q}_2)$ and $\boldsymbol\Lambda = \operatorname{diag}(6, 1)$, let $\mathbf{A} = \boldsymbol\Lambda^{-1/2}\mathbf{Q}^\top$. By (b),
# $$
# \mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top = \boldsymbol\Lambda^{-1/2}\mathbf{Q}^\top(\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top)\mathbf{Q}\boldsymbol\Lambda^{-1/2} = \boldsymbol\Lambda^{-1/2}\boldsymbol\Lambda\boldsymbol\Lambda^{-1/2} = \mathbf{I}.
# $$
# Explicitly, $\mathbf{A} = \begin{pmatrix} 2/\sqrt{30} & 1/\sqrt{30} \\ 1/\sqrt5 & -2/\sqrt5 \end{pmatrix} \approx \begin{pmatrix} 0.365 & 0.183 \\ 0.447 & -0.894 \end{pmatrix}$. The transformation rotates the axes onto the eigenvectors and divides each coordinate by its standard deviation: the elliptical cloud becomes a round one.
#
# (d) Let $\mathbf{z} = \mathbf{A}\mathbf{x}$; then $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\mathbf{x}^\top\boldsymbol\Sigma^{-1}\mathbf{x} = \mathbf{z}^\top\mathbf{z}$ (the squared Mahalanobis distance is the ordinary squared distance after whitening). For two standard Gaussian coordinates, $\mathbb{P}(\mathbf{z}^\top\mathbf{z} \le c^2) = 1 - e^{-c^2/2}$, so with $c^2 = 4$ the probability is $1 - e^{-2} \approx 0.865$.
#
# The cell below checks (a), (c) and (d) numerically.

# %%
Sigma = np.array([[5.0, 2.0], [2.0, 2.0]])
lam, Q = np.linalg.eigh(Sigma)                     # eigenvalues in increasing order
lam, Q = lam[::-1], Q[:, ::-1]                     # decreasing order, as in the solution above
Q = Q * np.sign(Q[0, :])                           # sign convention: first component positive
print("eigenvalues:", lam, "| eigenvectors (columns):\n", Q.round(4))

A_white = np.diag(lam ** -0.5) @ Q.T
print("\nwhitening matrix A:\n", A_white.round(4))
print("A Sigma A^T = I:", np.allclose(A_white @ Sigma @ A_white.T, np.eye(2)))

x = np.random.default_rng(SEED).multivariate_normal([0, 0], Sigma, size=200_000)
d2 = np.einsum("ij,jk,ik->i", x, np.linalg.inv(Sigma), x)
print(f"\nfraction with x^T Sigma^-1 x <= 4: {(d2 <= 4).mean():.4f}   theory 1 - exp(-2) = {1 - np.exp(-2):.4f}")
print("covariance of the whitened sample:\n", np.cov(x @ A_white.T, rowvar=False).round(3))

# %% [markdown]
# ### Python exercises

# %%
def solution_2_6():
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    frame, labels = iris.data, iris.target
    X, y = frame.to_numpy(), labels.to_numpy()

    # (a) class means: Boolean masks and a loop, then pandas
    means_mask = np.array([X[y == k].mean(axis=0) for k in range(3)])
    means_pandas = frame.groupby(labels).mean().to_numpy()
    print(pd.DataFrame(means_mask, index=iris.target_names, columns=frame.columns).round(3).to_string())
    print("agrees with pandas groupby:", np.allclose(means_mask, means_pandas))

    # (b) standardization with broadcasting
    Z = (X - X.mean(axis=0)) / X.std(axis=0)          # (150, 4) - (4,), then / (4,)
    print("\ncolumn means:", Z.mean(axis=0).round(10), "| column stds:", Z.std(axis=0).round(10))

    # (c) covariance of the standardized data = correlation matrix of the raw data
    cov_Z = np.cov(Z, rowvar=False, ddof=0)
    corr_X = np.corrcoef(X, rowvar=False)
    print("cov(standardized) equals corr(raw):", np.allclose(cov_Z, corr_X))
    print(pd.DataFrame(corr_X, index=frame.columns, columns=frame.columns).round(2).to_string())

solution_2_6()

# %% [markdown]
# (a) The loop over the species with a Boolean mask gives the same table as `groupby`. (b) Subtracting the vector of means and dividing by the vector of standard deviations broadcasts over the 150 rows; the standardized columns have mean 0 (up to rounding) and standard deviation 1 (`np.std` divides by $n$, the convention of `StandardScaler`). (c) The covariance of two standardized variables is $\operatorname{Cov}(x_i, x_j)/(\sigma_i\sigma_j)$, which is by definition their correlation $\rho_{ij}$: standardizing turns the covariance matrix into the correlation matrix. (Both must use the same divisor, which is why we pass `ddof=0`; with `ddof=1` the covariance matrix would be larger by the factor $n/(n-1)$.) The strong positive correlation between petal length and petal width, about 0.96, and the negative correlation between sepal width and the petal measurements are visible in the table.

# %%
def solution_2_7():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    def knn_predict(X_train, y_train, X_test, k=5):
        # squared distances between all test and training points, shape (n_test, n_train)
        sq_train, sq_test = (X_train ** 2).sum(axis=1), (X_test ** 2).sum(axis=1)
        D2 = sq_test[:, None] + sq_train[None, :] - 2 * X_test @ X_train.T
        nearest = np.argpartition(D2, k - 1, axis=1)[:, :k]          # indices of the k smallest distances in each row
        votes = y_train[nearest]                                      # labels of the neighbors, shape (n_test, k)
        return np.array([np.bincount(row).argmax() for row in votes])  # majority vote (ties go to the smallest label)

    X, y = load_wine(return_X_y=True)
    agreement, acc_mine, acc_sklearn = [], [], []
    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        scaler = StandardScaler().fit(X_tr)
        Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
        mine = knn_predict(Z_tr, y_tr, Z_te, k=5)
        reference = KNeighborsClassifier(n_neighbors=5).fit(Z_tr, y_tr).predict(Z_te)
        agreement.append((mine == reference).mean())
        acc_mine.append((mine == y_te).mean())
        acc_sklearn.append((reference == y_te).mean())

    print(f"agreement with scikit-learn over 30 splits: mean {np.mean(agreement):.4f}, minimum {np.min(agreement):.4f}")
    print(f"mean test accuracy: ours {np.mean(acc_mine):.3f} | scikit-learn {np.mean(acc_sklearn):.3f}")

solution_2_7()

# %% [markdown]
# The two classifiers make the same predictions on every test point of every split. The only loop is the one that counts the votes of each test point; the expensive part, the distances between all pairs, is a single matrix product (the Gram identity of Worked example 2.1). The accuracy of about 0.96 is the same as in Section 2.7.1, as it should be. With ties between distances or between votes the two implementations could in principle differ, but with continuous features they are rare.

# %%
def solution_2_8():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer(as_frame=True)
    df = bc.frame
    X, t = df.drop(columns="target"), df["target"]

    # (a) classes
    print("target names (index = code):", dict(enumerate(bc.target_names.tolist())))
    print("samples per class:", t.value_counts().sort_index().to_dict())

    # (b) standardized difference between the class means (pooled standard deviation)
    n0, n1 = (t == 0).sum(), (t == 1).sum()
    m0, m1 = X[t == 0].mean(), X[t == 1].mean()
    pooled = np.sqrt(((n0 - 1) * X[t == 0].var() + (n1 - 1) * X[t == 1].var()) / (n0 + n1 - 2))
    d = ((m0 - m1).abs() / pooled).sort_values(ascending=False)
    print("\nfive best separating features:")
    print(d.head(5).round(2).to_string())
    print("\nthree worst:")
    print(d.tail(3).round(2).to_string())

    # (c) scatter plot of the two best features
    f1, f2 = d.index[:2]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for code, color, marker in [(0, VERMILION, "o"), (1, BLUE, "s")]:
        sel = t == code
        ax.scatter(X.loc[sel, f1], X.loc[sel, f2], s=18, color=color, marker=marker, alpha=0.7, label=bc.target_names[code])
    ax.set(xlabel=f1, ylabel=f2, title="The two best separating features")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # (d) redundancy between two features
    print(f"\ncorrelation of 'mean radius' and 'mean perimeter': {df['mean radius'].corr(df['mean perimeter']):.4f}")

solution_2_8()

# %% [markdown]
# (a) The data set has 212 malignant tumors (code 0) and 357 benign ones (code 1). (b) The features describing the "worst" (largest) values of the concave points, the perimeter, and the radius, together with the mean concave points, separate the classes best: the class means differ by 2.5 to 2.7 pooled standard deviations, and the mean perimeter is fifth with 2.3. The worst features, the symmetry error, the texture error, and the mean fractal dimension, have a difference of at most 0.03 pooled standard deviations: the two classes look the same in these variables, taken one at a time. (c) In the scatter plot the two classes form two clouds with only a small overlap. (d) The correlation between `mean radius` and `mean perimeter` is 0.998: the perimeter of a nearly round tumor is just $2\pi$ times its radius, so the two columns carry the same information. A linear model with both features has almost linearly dependent columns (Exercise 1.10): the weights become unstable and hard to interpret, which is one of the reasons for the regularization of Chapter 6 and the feature selection of Chapter 10.

# %%
def solution_2_9():
    import matplotlib.pyplot as plt
    from scipy import stats

    rng = np.random.default_rng(SEED)
    reps, ns = 200_000, [1, 2, 3, 5, 10]
    uniform = rng.random((reps, max(ns)))                 # mean 1/2, standard deviation sqrt(1/12)
    exponential = rng.exponential(1.0, size=(reps, max(ns)))   # mean 1, standard deviation 1

    def ks_distance(samples, mu, sigma, n):
        z = (samples[:, :n].mean(axis=1) - mu) * np.sqrt(n) / sigma
        return stats.kstest(z, "norm").statistic

    ks_uniform = np.array([ks_distance(uniform, 0.5, np.sqrt(1 / 12), n) for n in ns])
    ks_expon = np.array([ks_distance(exponential, 1.0, 1.0, n) for n in ns])

    print(f"{'n':>3} {'uniform':>9} {'exponential':>12}")
    for n, a, b in zip(ns, ks_uniform, ks_expon):
        print(f"{n:>3} {a:>9.4f} {b:>12.4f}")
    slope_u = np.polyfit(np.log(ns), np.log(ks_uniform), 1)[0]
    slope_e = np.polyfit(np.log(ns), np.log(ks_expon), 1)[0]
    print(f"\nslope of log(KS) against log(n): uniform {slope_u:.2f}, exponential {slope_e:.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(ns, ks_uniform, "s-", color=BLUE, label="uniform (skewness 0)")
    ax.loglog(ns, ks_expon, "o--", color=ORANGE, label="exponential (skewness 2)")
    grid = np.array(ns, dtype=float)
    ax.loglog(grid, 0.16 / np.sqrt(grid), ":", color="gray", label=r"$\propto 1/\sqrt{n}$")
    ax.loglog(grid, 0.06 / grid, "-.", color="gray", label=r"$\propto 1/n$")
    ax.set(xlabel="n", ylabel="Kolmogorov-Smirnov distance to N(0, 1)", title="How fast is the central limit theorem?")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

solution_2_9()

# %% [markdown]
# The distance to the Gaussian shrinks for both distributions, but at different speeds: the slope on the log-log plot is about $-0.54$ for the exponential distribution (close to the $-1/2$ of an error proportional to $1/\sqrt{n}$) and about $-1.2$ for the uniform one, whose error shrinks like $1/n$ (the fitted slopes are estimates, and for the uniform at $n = 10$ the distance, about 0.003, is already close to the noise level of the experiment, of the order of $1/\sqrt{200{,}000}$). The explanation is the **skewness**. The leading error term in the central limit theorem is proportional to the skewness of the original distribution divided by $\sqrt{n}$. The exponential distribution is strongly skewed (skewness 2), so this term is present; the uniform distribution is symmetric (skewness 0), the term vanishes, and the next one, of order $1/n$, dominates. In practice: for symmetric, light-tailed data the Gaussian approximation of an average is excellent already with a handful of samples; for skewed or heavy-tailed data it needs many more.

# %%
def solution_2_10():
    from sklearn.datasets import load_wine
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    X, y = load_wine(return_X_y=True)
    seeds = range(30)

    # (a) scalers and k
    def mean_accuracy(make_model, k):
        scores = []
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            scores.append(make_model(k).fit(X_tr, y_tr).score(X_te, y_te))
        return np.mean(scores)

    makers = {
        "no scaling":      lambda k: KNeighborsClassifier(n_neighbors=k),
        "standardization": lambda k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
        "min-max":         lambda k: make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k)),
        "robust":          lambda k: make_pipeline(RobustScaler(), KNeighborsClassifier(n_neighbors=k)),
    }
    table = pd.DataFrame({name: {f"k = {k}": mean_accuracy(maker, k) for k in (1, 5, 15)} for name, maker in makers.items()})
    print("(a) mean test accuracy over 30 splits:")
    print(table.round(3).to_string())

    # (b) leakage through the scaler
    print("\n(b) scaling the whole data set before splitting (leak) against the pipeline:")
    X_scaled_all = StandardScaler().fit_transform(X)                  # the mistake: statistics of the test samples are used
    for k in (1, 5, 15):
        diffs, leak_scores, ok_scores = [], [], []
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            Z_tr, Z_te, _, _ = train_test_split(X_scaled_all, y, test_size=0.3, stratify=y, random_state=seed)
            leak = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr).score(Z_te, y_te)
            ok = makers["standardization"](k).fit(X_tr, y_tr).score(X_te, y_te)
            leak_scores.append(leak); ok_scores.append(ok); diffs.append(leak - ok)
        print(f"  k = {k:>2}: leak {np.mean(leak_scores):.4f} | pipeline {np.mean(ok_scores):.4f} | "
              f"difference {np.mean(diffs):+.4f} (std over splits {np.std(diffs):.4f})")

    # (c) leakage through label-based feature selection, on pure noise
    rng = np.random.default_rng(SEED)
    leak_scores, ok_scores = [], []
    for rep in range(30):
        X_noise = rng.normal(size=(60, 2000))
        y_noise = rng.integers(0, 2, size=60)                          # random labels: nothing can be learned
        X_sel = SelectKBest(f_classif, k=10).fit_transform(X_noise, y_noise)     # the mistake: selection sees all labels
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_noise, test_size=0.3, random_state=rep)
        leak_scores.append(KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr).score(X_te, y_te))
        X_tr, X_te, y_tr, y_te = train_test_split(X_noise, y_noise, test_size=0.3, random_state=rep)
        pipe = make_pipeline(SelectKBest(f_classif, k=10), KNeighborsClassifier(n_neighbors=5))
        ok_scores.append(pipe.fit(X_tr, y_tr).score(X_te, y_te))
    print("\n(c) pure noise, selection of 10 features out of 2000, 5-NN, 30 repetitions:")
    print(f"  selection before the split (leak): accuracy {np.mean(leak_scores):.3f} +/- {np.std(leak_scores):.3f}")
    print(f"  selection inside the pipeline    : accuracy {np.mean(ok_scores):.3f} +/- {np.std(ok_scores):.3f}")

solution_2_10()

# %% [markdown]
# (a) Without scaling the accuracy is about 0.70 to 0.73; with any of the three scalers it is about 0.95 to 0.97. Standardization and min-max scaling are practically equivalent on this data set (min-max is slightly better for $k = 15$), and the robust scaler is a bit behind. The best $k$ depends on the scaler, which is a first hint that $k$ must be tuned (Chapter 5).
#
# (b) Scaling the whole data set before the split gives results that differ from the correct pipeline by about $0.001$ or less on average, with a standard deviation over the splits of about 0.01 (for $k = 1$ the accuracies are even identical on every split): the difference is not systematic, it is lost in the noise. The reason is that the mean and standard deviation of 178 samples hardly change when 30% of the samples are left out: a scaler learns very little from the data. This is why leakage through scalers often goes unnoticed. It would be a mistake to conclude that the rule is unimportant.
#
# (c) On random labels no method can be better than chance, so the correct accuracy is 0.5, and the pipeline indeed gives about 0.52. The "leaky" procedure reaches about 0.82: among 2000 random features, some happen to correlate with the random labels *of the whole data set, test samples included*, and keeping the ten best of them builds the test labels into the features. The model seems to have found real structure and has found none. The more the preprocessing step *learns*, in particular when it uses the labels, the larger the leak: **every step that is fitted must be inside the pipeline.**

# %% [markdown]
# ## C.3 Chapter 3
#
# ### Manual exercises
#
# **Solution 3.1 — The Bayes classifier with unequal priors.**
#
# (a) With $\pi_0 = 0.8$, $\pi_1 = 0.2$, the Bayes classifier predicts class 1 when $\pi_1 p_1(x) > \pi_0 p_0(x)$:
# $$
# 0.2\,e^{-(x-2)^2/2} > 0.8\,e^{-x^2/2}
# \iff -\tfrac{(x-2)^2}{2} + \tfrac{x^2}{2} > \ln 4
# \iff 2x - 2 > \ln 4
# \iff x > 1 + \tfrac{\ln 4}{2} \approx 1.693.
# $$
# The threshold is **not** the midpoint $x = 1$: the rarer class must be more convincing, so the boundary moves toward its mean by $\sigma^2\ln(\pi_0/\pi_1)/(\mu_1 - \mu_0) = 0.693$.
#
# (b) With $t = 1 + \tfrac{\ln 4}{2}$, the two error probabilities are $\mathbb{P}(x > t \mid y = 0) = 1 - \Phi(t) = 0.0452$ and $\mathbb{P}(x \le t \mid y = 1) = \Phi(t - 2) = \Phi(-0.307) = 0.3795$, so
# $$
# R^\ast = 0.8 \cdot 0.0452 + 0.2 \cdot 0.3795 = 0.1121.
# $$
#
# (c) "Always predict 0" is wrong exactly on class 1: $R = 0.2$. The midpoint rule has $R = 0.8\,(1 - \Phi(1)) + 0.2\,\Phi(-1) = \Phi(-1) = 0.1587$ (the two error probabilities are equal, so the priors do not matter). The Bayes classifier, with $0.1121$, is better than both: ignoring the priors costs $0.1587 - 0.1121 = 0.0466$, and ignoring the input entirely costs $0.2 - 0.1121 = 0.0879$.
#
# **Solution 3.2 — Which prediction does each loss prefer?**
#
# (a) Add and subtract $\eta(\mathbf{x})$ inside the square:
# $$
# (y - f)^2 = (y - \eta)^2 + 2\,(y - \eta)(\eta - f) + (\eta - f)^2 .
# $$
# Condition on $\mathbf{x}$: $\eta(\mathbf{x})$ and $f(\mathbf{x})$ are constants, and $\mathbb{E}[y - \eta(\mathbf{x}) \mid \mathbf{x}] = 0$, so the expectation of the cross term is $\mathbb{E}\big[2\,(\eta - f)\,\mathbb{E}[y - \eta \mid \mathbf{x}]\big] = 0$. Taking the expectation of the other two terms gives the result. Its first term is $R^\ast$ and does not depend on $f$; the second is non-negative and zero only for $f = \eta$.
#
# (b) For $y$ uniform on $\{0, 1, 10\}$ the mean is $11/3 = 3.667$ and the variance is $101/3 - (11/3)^2 = 20.22$. The squared loss $\mathbb{E}[(y - a)^2] = 20.22 + (a - 3.667)^2$ is minimized by the **mean** $a = 3.667$. The expected absolute loss is piecewise linear in $a$ with kinks at the data values; its slope is $\tfrac13(\#\{y_i < a\} - \#\{y_i > a\})$, negative for $a < 1$ and positive for $a > 1$, so the minimum is at the **median** $a = 1$.
#
# | Constant | $\mathbb{E}[(y - a)^2]$ | $\mathbb{E}\lvert y - a\rvert$ |
# |---|---|---|
# | mean, $a = 3.667$ | **20.22** | 4.22 |
# | median, $a = 1$ | 27.33 | **3.33** |
#
# Each constant wins on its own loss. The outlier (the value 10) drags the mean away from the bulk of the data and contributes with its *square* to the squared loss; the median is not moved by how far the outlier is. This is the reason why the absolute loss is called *robust*, and why the choice of the loss is part of the specification of the problem.
#
# **Solution 3.3 — Sizing a test set.** Here $\varepsilon = 0.03$ and $\delta = 0.05$.
#
# (a) Hoeffding: $n \ge \ln(2/\delta)/(2\varepsilon^2) = \ln 40/0.0018 = 2049.4$, so **2050** examples.
#
# (b) Chebyshev, with $\sigma^2 \le \tfrac14$: $n \ge \sigma^2/(\varepsilon^2\delta) = 0.25/(0.0009 \cdot 0.05) = 5555.6$, so **5556** examples: 2.7 times more than Hoeffding.
#
# (c) With $p = 0.05$, $\sigma^2 = p(1 - p) = 0.0475$ and $n \ge 0.0475/(0.0009 \cdot 0.05) = 1055.6$, so **1056** examples, about half of the Hoeffding value. Hoeffding uses only the range $[0, 1]$ of the variable, and must hold for the *worst* variable of that range ($p = \tfrac12$, variance $\tfrac14$); Chebyshev uses the actual variance, which for a rare event is 5 times smaller. The ratio of the two sample sizes is $2\sigma^2/(\delta\ln(2/\delta))$, which is $0.51$ here. Chebyshev's dependence on $\delta$ is much worse ($1/\delta$ instead of $\ln(1/\delta)$), so for a smaller $\delta$ Hoeffding wins again: with $\delta = 10^{-4}$ it needs 5,502 examples against 527,778 for Chebyshev with the same variance $0.0475$. (Bernstein's inequality, not covered here, uses the variance *and* has the logarithmic dependence on $\delta$.)
#
# (d) For $m = 1000$: $\sqrt{\ln(40)/2000} = 0.043$, that is $\pm 4.3$ percentage points.
#
# **Solution 3.4 — The realizable case.**
#
# (a) Call a function $f$ **bad** if $R(f) > \varepsilon$. A bad function is consistent with one random example with probability $1 - R(f) < 1 - \varepsilon$, and with all $n$ independent examples with probability at most $(1 - \varepsilon)^n \le e^{-\varepsilon n}$ (because $1 - u \le e^{-u}$). The ERM output has zero training error, so if $R(\hat f) > \varepsilon$ some bad function is consistent with the whole sample. By the union bound over at most $\lvert\mathcal{H}\rvert$ bad functions, $\mathbb{P}(R(\hat f) > \varepsilon) \le \lvert\mathcal{H}\rvert\,e^{-\varepsilon n}$.
#
# (b) Setting $\lvert\mathcal{H}\rvert\,e^{-\varepsilon n} \le \delta$ gives $n \ge (\ln\lvert\mathcal{H}\rvert + \ln(1/\delta))/\varepsilon$. For the conjunctions, $\ln 3^{10} = 10.99$ and $\ln(1/0.01) = 4.61$, so $n \ge 15.59/0.05 = 311.8$: **312 examples**.
#
# (c) In the agnostic case Corollary 3.2 gives an estimation error of at most $2\varepsilon'$ with $\varepsilon' = \sqrt{(\ln\lvert\mathcal{H}\rvert + \ln(2/\delta))/(2n)}$; setting $2\varepsilon' = 0.05$ gives $n \ge 4\,(\ln\lvert\mathcal{H}\rvert + \ln(2/\delta))/(2 \cdot 0.05^2) = 4 \cdot 16.28/0.005 =$ **13,028 examples**, about 42 times more. The difference is in the dependence on the accuracy: **$1/\varepsilon$** when the training error can be zero, **$1/\varepsilon^2$** in general. In the realizable case a function with true risk $\varepsilon$ is eliminated very quickly, because a mistake on it is *certain* to appear in the sample if it is made often enough; in the agnostic case the empirical risks of good and almost good functions differ by amounts of order $\varepsilon$ that fluctuate with standard deviation of order $1/\sqrt{n}$, and telling them apart takes $n \sim 1/\varepsilon^2$.
#
# **Solution 3.5 — Why $\sqrt{\ln m}$?**
#
# (a) For any $s > 0$, Jensen's inequality for the convex function $z \mapsto e^{sz}$ gives
# $$
# e^{s\,\mathbb{E}[\max_j Z_j]} \le \mathbb{E}\big[e^{s\max_j Z_j}\big] = \mathbb{E}\big[\max_j e^{sZ_j}\big] \le \sum_{j=1}^m \mathbb{E}\big[e^{sZ_j}\big] \le m\,e^{s^2\sigma^2/2}.
# $$
# (The maximum of non-negative numbers is at most their sum; this is the union bound in disguise, and requires no independence.) Taking logarithms and dividing by $s$,
# $$
# \mathbb{E}\big[\max_j Z_j\big] \le \frac{\ln m}{s} + \frac{s\sigma^2}{2}.
# $$
# The right-hand side is minimized at $s = \sqrt{2\ln m}/\sigma$, where it equals $\sigma\sqrt{2\ln m}$. (For $m = 1$ the bound is $0$, which also follows from $s \to 0$.)
#
# (b) Let $Z_j = \mu - \hat{R}_j = \tfrac1n\sum_{i=1}^n(\mu - \ell_{ji})$ be an average of $n$ independent centered variables, where each term $(\mu - \ell_{ji})/n$ lies in an interval of length $1/n$. By Hoeffding's lemma each term has $\mathbb{E}[e^{s(\mu - \ell_{ji})/n}] \le e^{s^2/(8n^2)}$, and the product of the $n$ independent factors is $\mathbb{E}[e^{sZ_j}] \le e^{s^2/(8n)}$, which is the hypothesis of (a) with $\sigma^2 = 1/(4n)$. Hence
# $$
# \mathbb{E}\big[\mu - \min_j\hat{R}_j\big] = \mathbb{E}\big[\max_j Z_j\big] \le \sqrt{\tfrac{1}{4n}}\sqrt{2\ln m} = \sqrt{\frac{\ln m}{2n}},
# $$
# the thin curves of Figure 3.4. Observe that independence *across candidates* was never used: the bound holds even if the $m$ candidates are strongly correlated (in that case it is just less tight). Only the *number* of candidates enters, through a logarithm, which is why we can afford enormous finite classes: $10^6$ candidates cost only $\sqrt{13.8/(2n)}$.
#
# ### Python exercises

# %%
def solution_3_6():
    from scipy import stats

    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu0, mu1 = np.array([0.0, 0.0]), np.array([2.0, 1.0])

    # (a) direction and threshold of the Bayes classifier (equal priors, equal covariances)
    w = np.linalg.solve(Sigma, mu1 - mu0)
    threshold = w @ (mu0 + mu1) / 2
    print(f"(a) w = {w.round(4)}, threshold = {threshold:.4f}: predict class 1 when w'x > threshold")

    # (b) Bayes risk from the Mahalanobis distance
    delta = np.sqrt((mu1 - mu0) @ w)
    print(f"(b) Delta = {delta:.4f}, Bayes risk Phi(-Delta/2) = {stats.norm.cdf(-delta / 2):.4f}")

    # (c) Monte Carlo
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    y = rng.integers(0, 2, size=n)
    X = rng.multivariate_normal([0, 0], Sigma, size=n) + np.where(y[:, None] == 1, mu1, mu0)
    err_bayes = np.mean((X @ w > threshold) != y)
    print(f"(c) Monte Carlo risk of the Bayes classifier, n = 10^6: {err_bayes:.4f}")

    # (d) classifiers that use a single feature (threshold at the midpoint of the two means of that feature)
    for j in range(2):
        pred = X[:, j] > (mu0[j] + mu1[j]) / 2
        theory = stats.norm.cdf(-abs(mu1[j] - mu0[j]) / (2 * np.sqrt(Sigma[j, j])))
        print(f"(d) feature x{j + 1} alone: Monte Carlo {np.mean(pred != y):.4f}, theory Phi(-|dmu|/(2 sigma)) = {theory:.4f}")

    # Why x2 adds nothing: E[x2 | x1] is the same line in both classes
    slope = Sigma[0, 1] / Sigma[0, 0]
    for k, mu in enumerate((mu0, mu1)):
        print(f"E[x2 | x1, class {k}] = {mu[1] - slope * mu[0]:.1f} + {slope:.1f} x1")


solution_3_6()

# %% [markdown]
# (a) For equal covariances and equal priors the Bayes classifier is linear: $\mathbf{w} = \boldsymbol\Sigma^{-1}(\boldsymbol\mu_1 - \boldsymbol\mu_0) = (2, 0)^\top$ and the threshold is $\mathbf{w}^\top(\boldsymbol\mu_0 + \boldsymbol\mu_1)/2 = 2$: predict class 1 when $2x_1 > 2$, that is $x_1 > 1$. The classifier **does not use $x_2$ at all**. (b) $\Delta^2 = \mathbf{w}^\top(\boldsymbol\mu_1 - \boldsymbol\mu_0) = 4$, so $R^\ast = \Phi(-1) = 0.1587$, and (c) the Monte Carlo value $0.1590$ agrees within the sampling error (about $0.0004$ for $10^6$ points).
#
# (d) The feature $x_1$ alone reaches the Bayes risk, $0.1587$; the feature $x_2$ alone gives $\Phi(-0.5) = 0.3085$, which is much better than chance. Both facts are consistent: $x_2$ *is* informative by itself, because it is correlated with $x_1$ and the class shifts both means, but **given $x_1$ it adds nothing**, since the conditional mean $\mathbb{E}[x_2 \mid x_1] = 0.5\,x_1$ is the same line in the two classes. This is the first example of a principle that will return in feature selection (Chapter 10): the usefulness of a feature depends on which other features are already available.

# %%
def solution_3_7():
    rng = np.random.default_rng(SEED)
    sigma = 0.3

    def draw(n):
        x = rng.uniform(0, 1, n)
        return x, np.sin(2 * np.pi * x) + rng.normal(0, sigma, n)

    x_big, y_big = draw(100_000)
    x_test, y_test = draw(100_000)
    rows = []
    for m in (1, 3, 6):
        f_H = np.polyfit(x_big, y_big, m)
        risk_H = np.mean((y_test - np.polyval(f_H, x_test)) ** 2)
        approx = risk_H - sigma ** 2
        for n in (30, 300):
            risks = []
            for _ in range(300):
                x, y = draw(n)
                coef = np.polyfit(x, y, m)
                risks.append(np.mean((y_test - np.polyval(coef, x_test)) ** 2))
            rows.append({"degree m": m, "n": n, "R(f_H)": risk_H, "approximation": approx,
                         "E[R(f_S)]": np.mean(risks), "estimation": np.mean(risks) - risk_H,
                         "median R(f_S)": np.median(risks)})
    table = pd.DataFrame(rows)
    print(f"Bayes risk R* = {sigma ** 2:.4f}")
    print(table.round(4).to_string(index=False))


solution_3_7()

# %% [markdown]
# The approximation errors are $0.193$ for lines ($m = 1$, a sine period cannot be fitted by a line), $0.004$ for cubics and about zero for $m = 6$ (the value $-0.0001$ is a Monte Carlo error; the sine is almost exactly a degree-6 polynomial on $[0, 1]$). The estimation error is the opposite: it is largest for the richest class and shrinks with $n$.
#
# At $n = 30$ the best degree is $m = 3$ (mean risk $0.111$ against $0.304$ for $m = 1$ and $0.191$ for $m = 6$), at $n = 300$ it is $m = 6$ ($0.0921$ against $0.0956$ for $m = 3$): **the best complexity grows with the sample size**, as in Figure 3.3. The mean risk of $m = 6$ at $n = 30$ is much larger than its median ($0.191$ against $0.118$): a few of the 300 samples produce a polynomial that oscillates wildly near the edges of $[0, 1]$ and has an enormous risk. This is *variance* in its purest form, a rich class reacting to the idiosyncrasies of one small sample; a degree-9 polynomial fitted to 20 points has such an extreme mean risk, dominated by a handful of samples, that the mean is meaningless. Chapter 4 formalizes it with the bias–variance decomposition.

# %%
def solution_3_8():
    from itertools import product

    def dichotomies(points, rule_family):
        return len({tuple(rule(x) for x in points) for rule in rule_family(points)})

    def half_lines(points):                       # I[x > t]: thresholds between consecutive points
        cuts = np.concatenate(([points[0] - 1], (points[:-1] + points[1:]) / 2, [points[-1] + 1]))
        return [lambda x, t=t: int(x > t) for t in cuts]

    def interval_complements(points):             # I[x < a or x > b]
        cuts = np.concatenate(([points[0] - 1], (points[:-1] + points[1:]) / 2, [points[-1] + 1]))
        return [lambda x, a=a, b=b: int(x < a or x > b) for a in cuts for b in cuts if a <= b]

    print(" n | half-lines | n+1 | complements of intervals | n(n+1)/2+1")
    for n in range(1, 9):
        pts = np.arange(n, dtype=float)
        print(f"{n:>2} | {dichotomies(pts, half_lines):>10} | {n + 1:>3} | {dichotomies(pts, interval_complements):>24} | {n * (n + 1) // 2 + 1:>10}")

    # (b) sign(sin(omega x)) shatters x_j = 10^-j
    m = 6
    x = 10.0 ** -np.arange(1, m + 1)
    ok = 0
    for labels in product([-1, 1], repeat=m):
        b = (1 - np.array(labels)) / 2
        omega = np.pi * (1 + np.sum(b * 10.0 ** np.arange(1, m + 1)))
        ok += np.all(np.sign(np.sin(omega * x)) == np.array(labels))
    print(f"\nsign(sin(omega x)) reproduces {ok} of {2 ** m} labelings of the {m} points x_j = 10^-j")


solution_3_8()

# %% [markdown]
# (a) For $n$ distinct points on a line the half-lines produce $n + 1$ dichotomies (the threshold falls in one of the $n + 1$ gaps), so $\Pi(n) = n + 1$; since $\Pi(2) = 3 < 2^2$, no pair of points is shattered and $d_{\mathrm{VC}} = 1$. The complements of intervals produce $n(n + 1)/2 + 1$ dichotomies, exactly as the intervals do: the points labeled 0 form a contiguous block, which is empty or one of the $n(n+1)/2$ possible blocks. For $n = 2$ all four labelings are possible, but for $n = 3$ the labeling $(0, 1, 0)$ is not (its zeros are not contiguous), so $\Pi(3) = 7 < 8$ and $d_{\mathrm{VC}} = 2$. Both counts agree with the output above.
#
# (b) The table confirms that all $2^6 = 64$ labelings of $x_j = 10^{-j}$ are reproduced by $\mathrm{sign}(\sin(\omega x))$ with the prescribed $\omega$. The proof for general $m$ is the one outlined in the hint: for $i > j$ the term $\pi b_i 10^{i-j}$ is an even multiple of $\pi$ (it is $\pi$ times a multiple of 10) and does not change the sign of the sine; the term $i = j$ gives $\pi b_j$; the terms with $i < j$ add up to less than $\pi/5$ and the term $\pi 10^{-j}$ is smaller still. So the argument of the sine lies in $(\pi b_j, \pi b_j + \pi/2) \bmod 2\pi$, where the sign is $+$ for $b_j = 0$ and $-$ for $b_j = 1$. A class with a single parameter can therefore shatter arbitrarily large sets: the **number of parameters is not the capacity**.

# %%
def solution_3_9():
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    X30 = StandardScaler().fit_transform(load_breast_cancer().data)[rng.choice(569, 100, replace=False)]
    n = 100
    sigmas = rng.choice([-1.0, 1.0], size=(2000, n))
    rows = []
    for d in (30, 100, 300, 1000, 3000):
        X = np.hstack([X30, rng.normal(size=(n, d - 30))])
        S = sigmas @ X                                                   # 2000 x d: the sums sum_i sigma_i x_ij
        r2 = np.mean(np.linalg.norm(S, axis=1)) / n
        r1 = np.mean(np.max(np.abs(S), axis=1)) / n
        b2 = np.sqrt(np.sum(X ** 2)) / n
        b1 = np.max(np.linalg.norm(X, axis=0)) * np.sqrt(2 * np.log(2 * d)) / n
        rows.append({"d": d, "l2 ball": r2, "bound l2": b2, "l1 ball": r1, "bound l1": b1})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    print(f"\nl2 complexity grows by a factor {table['l2 ball'].iloc[-1] / table['l2 ball'].iloc[0]:.2f} from d = 30 to d = 3000 "
          f"(sqrt(3000/30) = {np.sqrt(100):.0f});")
    print(f"l1 complexity grows by a factor {table['l1 ball'].iloc[-1] / table['l1 ball'].iloc[0]:.2f}")


solution_3_9()

# %% [markdown]
# The $\ell_2$ complexity grows like $\sqrt{d}$ (a factor 10.5 from $d = 30$ to $d = 3000$, against $\sqrt{100} = 10$) and the bound $\sqrt{\sum_i\lVert\mathbf{x}_i\rVert^2}/n$ is nearly exact, because the squared norms of the rows are dominated by the $d - 30$ noise coordinates, whose squares add up to about $d$ per row. The point of Proposition 3.6 is that the bound depends on the *norm* of the data and not on the dimension: if the norm grows with $d$, as it does when we append noise columns, so does the bound, and **bounding $\lVert\mathbf{w}\rVert_2 \le 1$ does not protect against a large number of irrelevant features of order 1 each**. The $\ell_1$ complexity grows only by a factor of 2, as $\sqrt{\ln 2d}$ (the maximum over $2d$ signed columns, each of which behaves like a Gaussian with the same variance proxy) and the bound follows with a small slack (ratio between 1.4 and 2). This is the mathematical reason why $\ell_1$ regularization (Lasso, Chapter 10) works well when there are many irrelevant features, *provided that the true weight vector has a small $\ell_1$ norm*, that is, that it is sparse.

# %%
def solution_3_10():
    from scipy import stats

    rng = np.random.default_rng(SEED)
    z = stats.norm.ppf(0.975)
    reps = 20_000
    rows = []
    for p in (0.03, 0.2):
        for m in (20, 50, 200):
            K = rng.binomial(m, p, size=reps)
            ph = K / m
            hoeff = np.sqrt(np.log(2 / 0.05) / (2 * m)) * np.ones(reps)
            wald = z * np.sqrt(ph * (1 - ph) / m)
            centre = (ph + z ** 2 / (2 * m)) / (1 + z ** 2 / m)
            half_w = z * np.sqrt(ph * (1 - ph) / m + z ** 2 / (4 * m ** 2)) / (1 + z ** 2 / m)
            for name, lo, hi in [("Hoeffding", ph - hoeff, ph + hoeff), ("Wald", ph - wald, ph + wald),
                                 ("Wilson", centre - half_w, centre + half_w)]:
                rows.append({"p": p, "m": m, "interval": name, "coverage": np.mean((lo <= p) & (p <= hi)),
                             "mean width": np.mean(hi - lo)})
    table = pd.DataFrame(rows).pivot(index=["p", "m"], columns="interval", values=["coverage", "mean width"])
    print(table.round(3).to_string())


solution_3_10()

# %% [markdown]
# **Hoeffding's interval is valid but very conservative**: its coverage is 1.000 (or 0.999) in every row, much above the nominal 0.95, and it is two to five times wider than the others (a half-width of 0.30 for $m = 20$, which is useless for estimating an error rate of 0.03). **The Wald interval fails where it matters:** for a small error rate and a small test set its coverage is only 0.46 ($p = 0.03$, $m = 20$) and 0.78 ($m = 50$), because when $K = 0$ it collapses to a point and in general it relies on a normal approximation that is not accurate for $mp < 5$. For $p = 0.2$ and $m \ge 50$ it is fine. **The Wilson interval** has coverage close to 0.95 in all cases (0.937 to 0.979) and a width that is similar to the Wald interval where the latter works and larger where the Wald fails. It is the interval to report in practice; the Clopper–Pearson interval (`method="exact"`) is a conservative alternative that guarantees coverage at least 0.95 at the price of being slightly wider. Hoeffding's interval remains the right tool when a *guaranteed* statement without any approximation is needed, and as the formula to plan the size of a test set in advance.

# %% [markdown]
# ## C.4 Chapter 4
#
# ### Manual exercises
#
# **Solution 4.1 — k-NN by hand, with weights.** The distances from $x = 5$ to $1, 2, 4, 7, 9, 10$ are $4, 3, 1, 2, 4, 5$. Sorted: $x = 4$ (distance 1, label 1), $x = 7$ (2, label 1), $x = 2$ (3, label 0), then $x = 1$ and $x = 9$ **tied** at distance 4 (labels 0 and 0), and $x = 10$ (5, label 0).
#
# (a) $k = 1$: label 1. $k = 3$: the neighbors are $4, 7, 2$ with labels $1, 1, 0$: label **1**. $k = 5$: the neighbors are $4, 7, 2, 1, 9$ (the tie at distance 4 does not matter, because both tied points enter) with labels $1, 1, 0, 0, 0$: label **0**. The prediction changes with $k$.
#
# (b) Weighted votes with $k = 5$: class 1 receives $1/1 + 1/2 = 1.5$; class 0 receives $1/3 + 1/4 + 1/4 = 0.833$. The predicted class is **1**: the two close neighbors outweigh the three far ones, and the weighted vote is less sensitive to the choice of $k$ (it agrees with $k = 1$ and $k = 3$).
#
# (c) With $k = 6$ all the training points vote: the labels are $0, 0, 1, 1, 0, 0$, four zeros against two ones, so the vote gives 0 **for every query whatsoever**. It is a property of the method with $k = n$, not of the query: it ignores $x$ and returns the majority class of the training set, which is the rule of the smallest flexibility (the constant predictor of Chapter 1). If the classes had been split evenly, the answer would have depended on the tie-breaking rule of the implementation, as in Worked example 4.1.
#
# (d) The training error of 1-NN is **0**: each training point is its own nearest neighbor (distance 0). For 6-NN it is the proportion of points whose label differs from the training majority (0): $2/6 = 1/3$. The training error cannot be used to choose $k$.
#
# **Solution 4.2 — Bias and variance at the boundary and in the interior.** With $f(x) = x$ and the formulas of Section 4.5.2:
#
# (a) At $x_0 = 0$ the $k$ nearest neighbors are $x_1, \dots, x_k$ (that is, $0, 0.01, \dots, (k-1)/100$), whose average is $(k - 1)/200$. Hence the bias is $-(k - 1)/200$, the variance is $\sigma^2/k = 0.25/k$, and the expected squared error is
# $$
# 0.25 + \Big(\frac{k - 1}{200}\Big)^2 + \frac{0.25}{k}.
# $$
# For $k = 1, 5, 11, 21, 31, 51$ it is $0.500$, $0.300$, $0.275$, $0.272$, $0.281$, $0.317$.
#
# (b) At $x_0 = 0.5$ and odd $k$ the neighbors are symmetric around $x_0$, so the average of the linear function over them is exactly $f(x_0)$: the **bias is zero for every odd $k$**, and the error is $0.25 + 0.25/k$, which decreases with $k$ down to $0.25 + 0.25/101 = 0.2525$ for $k = 101$.
#
# (c) At the boundary the best $k$ minimizes $((k - 1)/200)^2 + 0.25/k$; the derivative vanishes where $k^2(k - 1) = 5000$, which is $k \approx 17$ (integer minimum at $k = 17$, with an excess error of $0.0211$). In the interior the best $k$ is the largest possible, because there is no bias to pay for. A single $k$ for the whole range is therefore a compromise: $k = 17$ is too small in the interior (excess error $0.25/17 = 0.0147$, against $0.0025$ with $k = 101$), and a large $k$ is damaging at the boundary ($k = 101$: the bias is $-0.5$, so bias$^2 = 0.25$ and the excess error is $0.2525$). This is one reason why local methods that adapt the neighborhood to the location, or correct the boundary bias by fitting a *local line* instead of a local constant, are preferred in practice.
#
# **Solution 4.3 — Cover–Hart with numbers.**
#
# (a) $R^\ast = \tfrac12\min\{0.9, 0.1\} + \tfrac12\min\{0.2, 0.8\} = \tfrac12 (0.1) + \tfrac12 (0.2) = \mathbf{0.15}$.
#
# (b) $R_{1\text{-NN}} = \tfrac12\cdot 2 \cdot 0.9 \cdot 0.1 + \tfrac12 \cdot 2 \cdot 0.2 \cdot 0.8 = 0.09 + 0.16 = \mathbf{0.25}$. The bound is $2R^\ast(1 - R^\ast) = 2 \cdot 0.15 \cdot 0.85 = 0.255$: the 1-NN risk is just below it, almost twice the Bayes risk (the ratio is $1.67$). The bound is nearly attained here because $r = \min(\eta, 1-\eta)$ takes the two values 0.1 and 0.2, which are close to each other; Jensen's inequality used in the proof is an equality when $r$ is constant.
#
# (c) For $\eta = 0.9$: $p_1 = 3(0.81) - 2(0.729) = 0.972$ and the error is $0.9(0.028) + 0.1(0.972) = 0.1224$. For $\eta = 0.2$: $p_1 = 3(0.04) - 2(0.008) = 0.104$ and the error is $0.2(0.896) + 0.8(0.104) = 0.2624$. Averaging, $R_{3\text{-NN}} = \tfrac12(0.1224 + 0.2624) = \mathbf{0.1924}$.
#
# (d) The gap between 1-NN and Bayes is $0.25 - 0.15 = 0.10$; with $k = 3$ it is $0.1924 - 0.15 = 0.0424$, so $k = 3$ closes $(0.25 - 0.1924)/0.10 = 58\%$ of the gap. Larger $k$ closes more, and the limit is the Bayes risk.
#
# **Solution 4.4 — Counting in high dimensions.**
#
# (a) $n s^d = 1$ with $s = 0.1$ gives $n = 10^d$: $10, 100, 10^5, 10^{10}$ and $10^{20}$ points for $d = 1, 2, 5, 10, 20$. (Ten billion points in dimension 10 just to have one neighbor within 10% of the range of every feature.)
#
# (b) $s = (k/n)^{1/d} = (10^{-5})^{1/d}$: $0.003$ for $d = 2$, $0.316$ for $d = 10$, $0.794$ for $d = 50$ and $0.891$ for $d = 100$. With a million points, in dimension 10 the cube that contains ten points spans 32% of each range; in dimension 100, 89%.
#
# (c) A point is *not* within $0.05$ of the boundary when all its coordinates lie in $[0.05, 0.95]$, which has probability $0.9^d$. The fraction near the boundary is $1 - 0.9^d$: $0.19$ for $d = 2$, $0.65$ for $d = 10$ and $0.995$ for $d = 50$. In high dimensions almost all the points are near the boundary of the cube, and the neighbors of a typical point are all on one side of it: the boundary effect of Figures 4.4 and 4.5, now everywhere.
#
# **Solution 4.5 — The best $k$ in $d$ dimensions.**
#
# (a) The bias at $\mathbf{x}_0$ is $\frac1k\sum_{i}(f(\mathbf{x}_0) - f(\mathbf{x}_i))$, and each term is at most $L\lVert\mathbf{x}_0 - \mathbf{x}_i\rVert \le Lr$ in absolute value, so $\text{bias}^2 \le L^2r^2$. The variance is $\sigma^2/k$ (Section 4.5.2). From $k = n v_d r^d$ we get $r = (k/(n v_d))^{1/d}$ and the bound $L^2\big(k/(nv_d)\big)^{2/d} + \sigma^2/k$.
#
# (b) Differentiating, $\frac{2}{d}L^2(nv_d)^{-2/d}k^{2/d - 1} - \sigma^2k^{-2} = 0$, i.e. $k^{(d+2)/d} = \frac{d\sigma^2}{2L^2}(nv_d)^{2/d}$, and raising to the power $d/(d+2)$ gives
# $$
# k^\ast = \Big(\frac{d\sigma^2}{2L^2}\Big)^{d/(d+2)}(nv_d)^{2/(d+2)} .
# $$
# At the optimum the two terms are of the same order (the variance term is $\sigma^2/k^\ast \propto n^{-2/(d+2)}$), so the minimal bound is proportional to $n^{-2/(d+2)}$.
#
# (c) To halve a bound that is $\propto n^{-2/(d+2)}$ the sample size must be multiplied by $2^{(d+2)/2}$: $2.8$ for $d = 1$, $4$ for $d = 2$, $11.3$ for $d = 5$ and $64$ for $d = 10$.
#
# (d) For $n = 10^4$, $L = \sigma = 1$: $k^\ast = 585$ for $d = 1$ (a fraction $0.058$ of the data), $43$ for $d = 5$ and $21$ for $d = 10$, with bounds $0.0026$, $0.081$ and $0.289$ on the excess error (for a noise variance of 1). In dimension 10, even with 10,000 points the best neighborhood contains only 21 points and has a radius $r = (21/(10^4 \cdot 2.55))^{1/10} = 0.48$, about half of the range of each feature: the Lipschitz assumption is then not enough to say that the function is nearly constant in the neighborhood, and the guarantee is poor. (The bound is an upper bound and real problems are often easier, as Digits shows; but the exponent $2/(d+2)$ cannot be improved without assumptions beyond Lipschitz continuity.)
#
# ### Python exercises

# %%
def solution_4_6():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    X, y = load_diabetes(return_X_y=True)

    def knn_regress(X_train, y_train, X_test, k):
        """k-NN regression with the Euclidean distance: average of the targets of the k nearest training points."""
        sq_train, sq_test = (X_train ** 2).sum(axis=1), (X_test ** 2).sum(axis=1)
        D2 = sq_test[:, None] + sq_train[None, :] - 2 * X_test @ X_train.T
        nearest = np.argpartition(D2, kth=k - 1, axis=1)[:, :k]
        return y_train[nearest].mean(axis=1)

    k, agree, mse_ours, mse_sk = 20, [], [], []
    for seed in range(30):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=seed)
        ours = knn_regress(Xa, ya, Xb, k)
        theirs = KNeighborsRegressor(n_neighbors=k).fit(Xa, ya).predict(Xb)
        agree.append(np.mean(np.isclose(ours, theirs)))
        mse_ours.append(np.mean((ours - yb) ** 2)); mse_sk.append(np.mean((theirs - yb) ** 2))
    print(f"fraction of identical predictions, averaged over 30 splits: {np.mean(agree):.4f} (minimum {np.min(agree):.4f})")
    print(f"mean test MSE, k = {k}: ours {np.mean(mse_ours):.1f} | scikit-learn {np.mean(mse_sk):.1f}")


solution_4_6()

# %% [markdown]
# The two implementations agree on **every** prediction of every split (the fraction of identical predictions is 1.0000), and the mean test MSE for $k = 20$ on the ten features is 3,310 on these 30 splits (the value 3,283 of Section 4.4 was obtained with 20 other splits). Regression has no vote, so the tie problem of classification is only relevant when the sum of the targets of the neighbors is affected by which of the tied points is chosen, which is a very rare event on continuous data.

# %%
def solution_4_7():
    import warnings

    rng = np.random.default_rng(SEED)
    f_step = lambda x: (x > 0.5).astype(float)
    sigma, n, n_sets = 0.3, 100, 2000
    points = np.array([0.2, 0.5])
    knn_ks, degrees = [1, 5, 10, 20, 50], [1, 3, 9]

    preds = {("k-NN", k): np.zeros((n_sets, 2)) for k in knn_ks}
    preds.update({("polynomial", m): np.zeros((n_sets, 2)) for m in degrees})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                   # polyfit warns about the conditioning of degree 9
        for s in range(n_sets):
            x = rng.uniform(0, 1, n)
            y = f_step(x) + rng.normal(0, sigma, n)
            order = np.argsort(np.abs(x[None, :] - points[:, None]), axis=1)
            for k in knn_ks:
                preds[("k-NN", k)][s] = y[order[:, :k]].mean(axis=1)
            for m in degrees:
                preds[("polynomial", m)][s] = np.polyval(np.polyfit(x, y, m), points)

    rows = []
    for (family, par), P in preds.items():
        for j, x0 in enumerate(points):
            b2 = (P[:, j].mean() - f_step(np.array([x0]))[0]) ** 2
            v = P[:, j].var()
            rows.append({"method": f"{family}, {'k' if family == 'k-NN' else 'degree'} = {par}", "x0": x0, "bias^2": b2, "variance": v, "expected error": sigma ** 2 + b2 + v})
    table = pd.DataFrame(rows).pivot(index="method", columns="x0", values=["bias^2", "variance", "expected error"])
    order_rows = [f"k-NN, k = {k}" for k in knn_ks] + [f"polynomial, degree = {m}" for m in degrees]
    print(table.loc[order_rows].round(4).to_string())
    print(f"\nnoise level sigma^2 = {sigma ** 2:.2f}")


solution_4_7()

# %% [markdown]
# **At $x_0 = 0.2$**, in the flat part of the function, k-NN has no bias until the neighborhood becomes wide enough to reach the jump ($k = 50$: bias$^2 = 0.0014$), so the expected error falls from $0.176$ at $k = 1$ (variance $0.086 \approx \sigma^2$) to a minimum of $0.094$ at $k = 20$, close to the noise floor $0.09$; for larger $k$ the bias starts to grow. Among the polynomials, degree 1 is best ($0.095$): a line is a good local approximation of a constant (bias$^2$ $0.0023$) and has small variance ($0.0029$), degree 3 has the largest bias at this point ($0.0118$, the cubic oscillates to approximate the step) and degree 9 a lower bias ($0.0022$) but the largest variance ($0.0079$).
#
# **At $x_0 = 0.5$**, exactly at the jump, *every* method has bias$^2 \approx 0.25$: a smooth predictor, or an average over neighbors on both sides of the jump, returns about $0.5$ while $f(0.5) = 0$, and the bias is $0.5$ whatever the method. Nothing can be done about it with this kind of method (a discontinuity is not "smooth"), so the ranking is decided by the variance, and it is *inverted* with respect to the previous point: the larger the $k$, the better ($0.684$ at $k = 1$, $0.347$ at $k = 50$), and the degree-1 polynomial, with the smallest variance, is best overall ($0.341$). The same method is therefore best or worst depending on the location: the optimal degree of smoothing depends on where we are in the input space, a fact that adaptive methods exploit (Chapter 11).

# %%
def solution_4_8():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    X, y = load_diabetes(return_X_y=True)
    x = X[:, 2]                                                           # bmi

    def nadaraya_watson(x_train, y_train, x_test, h):
        d2 = (x_test[:, None] - x_train[None, :]) ** 2
        w = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (2 * h * h))    # subtracting the smallest d2 avoids 0/0 far from the data
        return (w * y_train).sum(axis=1) / w.sum(axis=1)

    hs = np.array([0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048, 0.064, 0.1])
    ks = [10, 20, 30, 40, 60, 80]
    mse_h, mse_k = np.zeros((20, len(hs))), np.zeros((20, len(ks)))
    for r in range(20):
        xa, xb, ya, yb = train_test_split(x, y, test_size=0.3, random_state=r)
        for j, h in enumerate(hs):
            mse_h[r, j] = np.mean((nadaraya_watson(xa, ya, xb, h) - yb) ** 2)
        for j, k in enumerate(ks):
            mse_k[r, j] = np.mean((KNeighborsRegressor(n_neighbors=k).fit(xa[:, None], ya).predict(xb[:, None]) - yb) ** 2)
    print("kernel regression, test MSE by bandwidth h:", {float(h): int(round(v)) for h, v in zip(hs, mse_h.mean(axis=0))})
    print("k-NN regression, test MSE by k           :", {k: int(round(v)) for k, v in zip(ks, mse_k.mean(axis=0))})
    h_best, k_best = hs[mse_h.mean(axis=0).argmin()], ks[mse_k.mean(axis=0).argmin()]
    print(f"best: h = {h_best} (MSE {mse_h.mean(axis=0).min():.0f}) | k = {k_best} (MSE {mse_k.mean(axis=0).min():.0f})")

    grid = np.linspace(x.min(), x.max(), 400)
    fit_h = nadaraya_watson(x, y, grid, h_best)
    fit_k = KNeighborsRegressor(n_neighbors=k_best).fit(x[:, None], y).predict(grid[:, None])
    for x_probe in (-0.05, 0.0, 0.1, 0.15):
        in_window = int(np.sum(np.abs(x - x_probe) <= h_best))
        print(f"x = {x_probe:+.2f}: points within +/- h: {in_window:>3} | radius of the {k_best}-NN neighborhood: {np.sort(np.abs(x - x_probe))[k_best - 1]:.3f}"
              f" | fits: kernel {nadaraya_watson(x, y, np.array([x_probe]), h_best)[0]:.0f}, k-NN {fit_k[np.argmin(np.abs(grid - x_probe))]:.0f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(x, y, s=12, color="gray", alpha=0.4, label="patients")
    ax.plot(grid, fit_h, color=VERMILION, linewidth=2.2, label=f"kernel regression, h = {h_best}")
    ax.plot(grid, fit_k, color=BLUE, linewidth=2.2, linestyle="--", label=f"k-NN, k = {k_best}")
    ax.set(xlabel="body-mass index (centered and scaled)", ylabel="disease progression", title="Fixed bandwidth against fixed number of neighbors")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    plt.show()


solution_4_8()

# %% [markdown]
# With a bandwidth of $h = 0.016$ the kernel regression has a test MSE of 3,863, against 3,966 for the best k-NN ($k = 40$): about 3% lower, a difference that these 20 splits cannot establish as significant. The figure and the last lines of the output show where the two methods differ. In the **dense** region ($x = -0.05$ and $x = 0$: 88 and 112 points in the window $\pm h$) the two fits coincide (108 against 107, and 150 against 147), because both average a few dozen nearby points. In the **sparse** region ($x = 0.15$) only 2 patients fall within $\pm h$: the kernel estimate (286) is determined by two points and is therefore noisy (the wiggle at the right end of the figure), while k-NN is forced to average 40 patients, reaching out to a distance of $0.082$, which gives a flat, smoother and more biased estimate (248; all its neighbors lie on one side). This is the essential difference: **a fixed bandwidth has the same bias everywhere and a variance that depends on the local density, while a fixed $k$ has the same variance everywhere and a bias that grows where the data are sparse.** (Kernel regression also gives a smooth curve, whereas k-NN gives a staircase.) Both are special cases of local averaging with weights that decrease with the distance, and Chapter 9 will show kernels in a different role.

# %%
def solution_4_9():
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    rows = []
    for m in (100, 1000):
        res = {"5-NN": [], "selection + 5-NN (pipeline)": [], "logistic regression": [], "selection BEFORE the split + 5-NN (leak)": []}
        kept = []
        for r in range(15):
            Z = np.hstack([X, np.random.default_rng(r).normal(size=(len(X), m))])
            Za, Zb, ya, yb = train_test_split(Z, y, test_size=0.3, random_state=r, stratify=y)
            res["5-NN"].append(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(Za, ya).score(Zb, yb))
            pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=30), KNeighborsClassifier(n_neighbors=5)).fit(Za, ya)
            res["selection + 5-NN (pipeline)"].append(pipe.score(Zb, yb))
            kept.append(pipe[1].get_support()[:30].sum())
            res["logistic regression"].append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Za, ya).score(Zb, yb))
            # the mistake: the selection sees all the labels, test samples included
            Zs = SelectKBest(f_classif, k=30).fit_transform(StandardScaler().fit_transform(Z), y)
            Zsa, Zsb, ysa, ysb = train_test_split(Zs, y, test_size=0.3, random_state=r, stratify=y)
            res["selection BEFORE the split + 5-NN (leak)"].append(KNeighborsClassifier(n_neighbors=5).fit(Zsa, ysa).score(Zsb, ysb))
        print(f"\n{m} noise columns:")
        for name, v in res.items():
            print(f"  {name:<42}{np.mean(v):.4f}  (std {np.std(v):.4f})")
        print(f"  original features among the 30 selected: {np.mean(kept):.1f} of 30")


solution_4_9()

# %% [markdown]
# With 100 noise columns the plain 5-NN has an accuracy of 0.910; with the selection step inside the pipeline it recovers 0.955, better than logistic regression (0.945); with 1,000 noise columns the numbers are 0.803 and 0.957 against 0.923. On average 25 of the 30 selected features are original ones, out of 100 or 1,000 candidates that are mostly noise: the F-statistic of Chapter 10 finds the informative features easily because they carry a strong signal. The scores of 5-NN are almost the same with 100 and with 1,000 noise columns *after the selection*, which shows that it is the number of *irrelevant features that reach the distance* that damages k-NN, not the number of columns in the table.
#
# (c) The version with the selection done **before** the split (the leak of Section 2.7) gives 0.956 and 0.954: practically the same as the correct pipeline, and well inside the standard deviation over the splits (0.012 to 0.015). Here the leak is harmless because the signal is real and strong: the same 25 or so original features would be selected from the training set alone. As in Exercise 2.10, the leak becomes dangerous when the selection can exploit *chance* correlations, that is when there are many candidate features and few samples or no real signal; the pipeline is the safe habit in every case, because one cannot know beforehand which case one is in.

# %%
def solution_4_10():
    from scipy import stats
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(SEED)
    bayes = stats.norm.cdf(-1)
    sizes = [30, 100, 300, 1000, 3000, 10000]
    rules = {"k ≈ √n": lambda n: min(int(round(np.sqrt(n))) | 1, n), "k ≈ n^(2/3)": lambda n: min(int(round(n ** (2 / 3))) | 1, n)}

    def draw(n, d):
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, d))
        X[:, 0] += 2 * y
        return X, y

    results = {}
    for d in (2, 10):
        X_fresh, y_fresh = draw(20_000, d)
        rows = []
        for n in sizes:
            errs = {name: [] for name in rules}
            for _ in range(6):
                X_s, y_s = draw(n, d)
                for name, k_of_n in rules.items():
                    model = KNeighborsClassifier(n_neighbors=k_of_n(n), algorithm="brute").fit(X_s, y_s)
                    errs[name].append(np.mean(model.predict(X_fresh) != y_fresh))
            rows.append({"n": n, **{name: np.mean(v) for name, v in errs.items()}})
        results[d] = pd.DataFrame(rows).set_index("n")
        print(f"d = {d}   (Bayes risk {bayes:.4f})")
        print(results[d].round(4).to_string())
        for name in rules:
            ok = results[d].index[(results[d][name] - bayes) <= 0.01]
            print(f"   smallest n within 0.01 of the Bayes risk, {name}: {ok.min() if len(ok) else 'none in the grid'}")
        print()

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for d, (color, marker) in {2: (BLUE, "o"), 10: (VERMILION, "s")}.items():
        for name, ls in zip(rules, ["-", "--"]):
            ax.plot(sizes, results[d][name], color=color, marker=marker, linestyle=ls, linewidth=2, label=f"d = {d}, {name}")
    ax.axhline(bayes, color="black", linewidth=1.4)
    ax.text(32, bayes + 0.004, "Bayes risk", fontsize=9.5)
    ax.set_xscale("log")
    ax.set(xlabel="training-set size $n$", ylabel="test error", title="Consistency is slower in higher dimension (simulated data)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


solution_4_10()

# %% [markdown]
# In dimension 2 the error is within 0.01 of the Bayes risk (0.1587) from $n = 100$ on for both rules, and reaches 0.157 at $n = 10{,}000$, as in Figure 4.8. In dimension 10, **with exactly the same Bayes risk and with only the first coordinate informative**, the convergence is much slower: at $n = 100$ the errors are 0.213 and 0.205 (against 0.168 and 0.165 in dimension 2), and the smallest $n$ of the grid within 0.01 of the Bayes risk is 3,000 for $k \approx \sqrt{n}$ and 1,000 for $k \approx n^{2/3}$, 30 and 10 times larger than in dimension 2. Nine useless coordinates make the distances noisy: two points of the same class are as far apart as two of different classes unless $n$ is large enough for the nearest neighbors to be close in all ten coordinates at once, which is the third face of the curse (Section 4.7.1). Note also that the larger $k$ is better in dimension 10 at every $n$ of the grid: when the individual neighbors are less informative, averaging more of them pays. (Exercise 4.9 shows how to reduce the penalty, by removing the useless coordinates.)

# %% [markdown]
# ## C.5 Chapter 5
#
# ### Manual exercises
#
# **Solution 5.1 — How much can a validation set tell?**
#
# (a) $\hat p = 0.87$, so $\operatorname{SE} = \sqrt{0.87 \cdot 0.13/100} = 0.0336$ and the 95% interval is $0.87 \pm 1.96 \cdot 0.0336 = 0.87 \pm 0.066$, that is **[0.804, 0.936]**: a validation set of 100 examples cannot distinguish an 81% model from a 93% one.
#
# (b) We need $1.96\sqrt{p(1-p)/m} \le 0.02$, that is $m \ge 1.96^2\,p(1-p)/0.02^2$. For $p = 0.9$: $m \ge 3.8416 \cdot 0.09/0.0004 = 864.4$, so **865 examples**. In the worst case $p = \tfrac12$, $p(1-p) = 0.25$ and $m \ge 2{,}401$.
#
# (c) Hoeffding's inequality (Section 3.10.2) asks for $m \ge \ln(2/0.05)/(2 \cdot 0.02^2) = 4611.1$, that is **4,612 examples**: about twice the worst-case number of the normal approximation, and 5.3 times the number for $p = 0.9$. Hoeffding uses only the fact that the loss is in $[0, 1]$, so it holds for every $m$ and every $p$ without any approximation, but it ignores the variance $p(1-p)$, which is what makes the normal interval narrower when the accuracy is high. (The Wilson interval of Section 3.10.2 gets the best of both: it is valid for small samples and uses the variance.)
#
# (d) The first model is right on $85 + 5 = 90$ examples and the second on $85 + 2 = 87$. Only the $b + c = 7$ discordant examples carry information: if the models had the same error rate, $b \sim \text{Binomial}(7, \tfrac12)$, and
# $$
# p = 2\,\mathbb{P}\big(\text{Bin}(7, \tfrac12) \le 2\big) = 2\cdot\frac{1 + 7 + 21}{128} = \frac{58}{128} = \mathbf{0.453}.
# $$
# The unpaired comparison of two accuracies would give $z = 0.03/\sqrt{0.09/100 + 0.1131/100} = 0.67$, that is $p \approx 0.51$. Pairing helps, but here only slightly: with seven discordant examples the data simply contain little evidence about the difference.
#
# **Solution 5.2 — Counting model fits.**
#
# (a) Each of the $5 \times 4 = 20$ configurations is evaluated with $5 \times 3 = 15$ fits, so $20 \cdot 15 = 300$, plus one refit of the winner: **301 fits**.
#
# (b) Each outer fold runs the whole search on its training part (301 fits, including the refit of its winner, which is the model scored on the outer held-out fold): $5 \times 301 = \mathbf{1{,}505}$ fits.
#
# (c) Random search with 20 draws evaluates 20 configurations, exactly as many as the grid: $20 \cdot 15 + 1 = \mathbf{301}$ fits. (Random search saves nothing here; its advantage is that a *smaller* number of draws, say 8 or 10, often does as well as the full grid; see Exercise 5.8.)
#
# (d) Leave-one-out needs $n = 1000$ fits per configuration: $12 \cdot 1000 + 1 = \mathbf{12{,}001}$ fits.
#
# For a model that takes one minute to fit these are 301 minutes (five hours), 1,505 minutes (25 hours) and 12,001 minutes (more than eight days). The design must change: fewer folds or no repeats, a smaller or random search, successive halving, a faster surrogate model for the search, parallel execution, or a single validation split instead of cross-validation.
#
# **Solution 5.3 — ROC and AUC by hand.** Sorted by decreasing score, the examples are $0.9^{+}, 0.7^{-}, 0.6^{+}, 0.4^{+}, 0.4^{-}, 0.2^{-}, 0.1^{-}$, with $n_+ = 3$ and $n_- = 4$.
#
# (a) Lowering the threshold past each block of scores, the curve moves up by $1/3$ for each positive and right by $1/4$ for each negative; the two tied examples at 0.4 are passed together, a diagonal step. The points $(\text{FPR}, \text{TPR})$ are
# $$
# (0, 0),\ (0, \tfrac13),\ (\tfrac14, \tfrac13),\ (\tfrac14, \tfrac23),\ (\tfrac12, 1),\ (\tfrac34, 1),\ (1, 1).
# $$
#
# (b) Trapezoid rule: the vertical steps have zero width; the horizontal step from $(0, \tfrac13)$ to $(\tfrac14, \tfrac13)$ contributes $\tfrac14\cdot\tfrac13 = \tfrac1{12}$; the diagonal step from $(\tfrac14, \tfrac23)$ to $(\tfrac12, 1)$ contributes $\tfrac14\cdot\tfrac{(2/3 + 1)}{2} = \tfrac{5}{24}$; the last two steps contribute $\tfrac14 + \tfrac14 = \tfrac12$. The total is $\tfrac{2}{24} + \tfrac{5}{24} + \tfrac{12}{24} = \tfrac{19}{24} = \mathbf{0.7917}$. Counting pairs: the positive 0.9 beats all 4 negatives; the positive 0.6 beats 3 (0.4, 0.2, 0.1) and loses to 0.7; the positive 0.4 beats 2 (0.2, 0.1), loses to 0.7 and ties with the other 0.4, which counts $\tfrac12$. The total is $4 + 3 + 2.5 = 9.5$ pairs of 12, and $9.5/12 = \tfrac{19}{24}$: **the two computations agree**, as Proposition 5.1 says.
#
# (c) Predicting positive when the score is at least $t$:
#
# | $t$ | TP | FP | precision | recall | $F_1$ |
# |---|---|---|---|---|---|
# | 0.9 | 1 | 0 | 1 | 1/3 | 0.500 |
# | 0.7 | 1 | 1 | 1/2 | 1/3 | 0.400 |
# | 0.6 | 2 | 1 | 2/3 | 2/3 | 0.667 |
# | 0.4 | 3 | 2 | 3/5 | 1 | **0.750** |
# | 0.2 | 3 | 3 | 1/2 | 1 | 0.667 |
# | 0.1 | 3 | 4 | 3/7 | 1 | 0.600 |
#
# The $F_1$ score is maximal at $t = 0.4$, with $F_1 = 0.75$ (precision 0.6, recall 1). Note that the tied scores force us to flag both examples at 0.4 or neither.
#
# **Solution 5.4 — The cost-optimal threshold.**
#
# (a) Predicting positive costs, in expectation, $(1 - \hat p)\,c_{FP}$ (we are wrong when $y = 0$); predicting negative costs $\hat p\,c_{FN}$. We predict positive when $(1-\hat p)\,c_{FP} \le \hat p\,c_{FN}$, that is $c_{FP} \le \hat p\,(c_{FP} + c_{FN})$, which is $\hat p \ge t^\ast = c_{FP}/(c_{FP} + c_{FN})$.
#
# (b) $t^\ast = 5/(5 + 200) = \mathbf{0.0244}$: flag a transaction as soon as the estimated probability of fraud exceeds 2.4%.
#
# (c) Of 10,000 transactions, 50 are fraudulent and 9,950 are legitimate. TPR 0.90 gives $TP = 45$, $FN = 5$; FPR 0.02 gives $FP = 199$, $TN = 9{,}751$. Precision $= 45/(45 + 199) = \mathbf{0.184}$. The cost is $200 \cdot 5 + 5 \cdot 199 = 1{,}995$, that is **0.1995 per transaction**. "Flag nothing" costs $200 \cdot 50 = 10{,}000$ (1.0 per transaction) and "flag everything" costs $5 \cdot 9{,}950 = 49{,}750$ (4.975 per transaction). The detector cuts the cost by a factor of five although only 18% of its alarms are real: a low precision is perfectly acceptable when a miss costs forty times a false alarm.
#
# (d) It does not change: $t^\ast$ depends only on the **ratio** $c_{FN}/c_{FP}$, which here stays at 40. Doubling both costs doubles the total cost but leaves the optimal decision unchanged.
#
# **Solution 5.5 — Leave-one-out for least squares without refitting.** Write $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$, so that $\hat{\boldsymbol\beta} = \mathbf{A}^{-1}\mathbf{X}^\top\mathbf{y}$ and $h_{ii} = \mathbf{x}_i^\top\mathbf{A}^{-1}\mathbf{x}_i$.
#
# (a) Without example $i$, $\mathbf{X}_{(i)}^\top\mathbf{X}_{(i)} = \mathbf{A} - \mathbf{x}_i\mathbf{x}_i^\top$ and $\mathbf{X}_{(i)}^\top\mathbf{y}_{(i)} = \mathbf{X}^\top\mathbf{y} - \mathbf{x}_i y_i$. By the Sherman–Morrison formula, $(\mathbf{A} - \mathbf{x}_i\mathbf{x}_i^\top)^{-1} = \mathbf{A}^{-1} + \dfrac{\mathbf{A}^{-1}\mathbf{x}_i\mathbf{x}_i^\top\mathbf{A}^{-1}}{1 - h_{ii}}$. Therefore
# $$
# \hat{\boldsymbol\beta}_{(i)} = \Big(\mathbf{A}^{-1} + \frac{\mathbf{A}^{-1}\mathbf{x}_i\mathbf{x}_i^\top\mathbf{A}^{-1}}{1 - h_{ii}}\Big)\big(\mathbf{X}^\top\mathbf{y} - \mathbf{x}_i y_i\big)
# = \hat{\boldsymbol\beta} - \mathbf{A}^{-1}\mathbf{x}_i y_i + \frac{\mathbf{A}^{-1}\mathbf{x}_i\,\hat y_i}{1 - h_{ii}} - \frac{\mathbf{A}^{-1}\mathbf{x}_i\,h_{ii}\,y_i}{1 - h_{ii}},
# $$
# using $\mathbf{x}_i^\top\mathbf{A}^{-1}\mathbf{X}^\top\mathbf{y} = \mathbf{x}_i^\top\hat{\boldsymbol\beta} = \hat y_i$ and $\mathbf{x}_i^\top\mathbf{A}^{-1}\mathbf{x}_i = h_{ii}$. Collecting the terms in $y_i$: $-y_i\big(1 + \tfrac{h_{ii}}{1 - h_{ii}}\big) = -\tfrac{y_i}{1 - h_{ii}}$, so
# $$
# \hat{\boldsymbol\beta}_{(i)} = \hat{\boldsymbol\beta} - \mathbf{A}^{-1}\mathbf{x}_i\,\frac{y_i - \hat y_i}{1 - h_{ii}} = \hat{\boldsymbol\beta} - \mathbf{A}^{-1}\mathbf{x}_i\,\frac{e_i}{1 - h_{ii}}.
# $$
# The prediction for example $i$ from the model that did not see it is $\mathbf{x}_i^\top\hat{\boldsymbol\beta}_{(i)} = \hat y_i - h_{ii}\,\dfrac{e_i}{1 - h_{ii}}$, and the leave-one-out residual is
# $$
# y_i - \mathbf{x}_i^\top\hat{\boldsymbol\beta}_{(i)} = e_i + \frac{h_{ii}\,e_i}{1 - h_{ii}} = \frac{e_i}{1 - h_{ii}}. \qquad\square
# $$
#
# (b) The leave-one-out mean squared error is $\dfrac{1}{n}\sum_{i=1}^n \Big(\dfrac{e_i}{1 - h_{ii}}\Big)^2$ (the PRESS statistic divided by $n$): it needs the residuals and the diagonal of the hat matrix of **one** fit on all the data, instead of $n$ fits. (The same trick works for every *linear smoother*, that is for every method whose fitted values are $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ for a matrix $\mathbf{H}$ that does not depend on $\mathbf{y}$, including ridge regression.)
#
# (c) The leverages satisfy $0 \le h_{ii} \le 1$ and $\sum_i h_{ii} = d$. An example with $h_{ii}$ close to 1 is far from the others in the input space: the fit passes almost through it, so its ordinary residual $e_i$ is small, but without it the model has nothing to extrapolate from, and the leave-one-out residual $e_i/(1 - h_{ii})$ can be large. High-leverage examples are exactly where the training residuals are most misleading about the prediction error.
#
# ### Python exercises

# %%
def solution_5_6():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def stratified_kfold_splits(y, K, seed):
        """Shuffle the indices of each class and deal them out to the K folds in turn."""
        rng = np.random.default_rng(seed)
        folds = [[] for _ in range(K)]
        for cls in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == cls))
            for j, chunk in enumerate(np.array_split(idx, K)):
                folds[j].extend(chunk.tolist())
        folds = [np.array(sorted(f)) for f in folds]
        return [(np.setdiff1d(np.arange(len(y)), held), held) for held in folds]

    def shuffled_kfold_splits(n, K, seed):
        folds = np.array_split(np.random.default_rng(seed).permutation(n), K)
        return [(np.setdiff1d(np.arange(n), held), held) for held in folds]

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y                                               # 1 = malignant
    rng = np.random.default_rng(SEED)
    keep = np.r_[np.flatnonzero(y == 0), rng.choice(np.flatnonzero(y == 1), size=20, replace=False)]
    Xs, ys = X[keep], y[keep]
    print(f"imbalanced data set: {len(ys)} patients, {int(ys.sum())} malignant ({ys.mean():.3f})")

    # (b) the class counts in each fold differ by at most one from the ideal proportion
    worst = 0
    for seed in range(50):
        for _, held in stratified_kfold_splits(ys, 5, seed):
            for cls in (0, 1):
                ideal = np.mean(ys == cls) * len(held)
                worst = max(worst, abs(np.sum(ys[held] == cls) - ideal))
    print(f"largest deviation from the ideal class count in a fold, over 50 seeds: {worst:.2f} examples")

    # (c) plain shuffled folds against stratified folds
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    out = {}
    for name, make in {"shuffled": lambda s: shuffled_kfold_splits(len(ys), 5, s),
                       "stratified": lambda s: stratified_kfold_splits(ys, 5, s)}.items():
        acc, min_pos = [], []
        for seed in range(50):
            splits = make(seed)
            acc.append(cross_val_score(model, Xs, ys, cv=splits).mean())
            min_pos.append(min(int(ys[held].sum()) for _, held in splits))
        out[name] = {"mean accuracy": np.mean(acc), "sd across seeds": np.std(acc, ddof=1),
                     "fewest malignant in a fold": int(np.min(min_pos)), "mean of that minimum": np.mean(min_pos)}
    table = pd.DataFrame(out).T.astype({"fewest malignant in a fold": int})
    print(table.round(4).to_string())


solution_5_6()

# %% [markdown]
# **(b)** In all 50 stratified 5-fold splits (250 folds) the number of examples of each class in a fold is within 0.03 of the ideal proportion. With 20 malignant examples and 5 folds every fold receives exactly 4 of them, and the 357 benign ones are dealt out as 72, 72, 71, 71, 71, so the deviation from the ideal count is only due to the fact that the folds themselves have 75 or 76 examples.
#
# **(c)** Stratification does **not** change the accuracy estimate: the mean accuracy is 0.978 for shuffled folds and 0.979 for stratified folds, and the standard deviation across seeds is 0.0022 against 0.0026, a difference that 50 seeds cannot resolve. What it changes is the **composition of the folds**. With shuffled folds there is, over the 50 seeds, at least one split in which a fold contains no malignant tumor at all (the smallest number found is 0), and the smallest fold contains on average 1.86 malignant examples; with stratified folds the smallest fold always contains 4. On a data set with 5% positives the accuracy hides the problem, because it is dominated by the benign class, but any metric that is computed *per fold* and involves the positive class (recall, precision, AUC) is undefined for a fold without positives and extremely noisy for a fold with one or two. Stratification costs nothing and removes the problem, which is why it is the default for classification in scikit-learn.

# %%
def solution_5_7():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    Ks = [2, 5, 10, 20]
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(100):
        perm = rng.permutation(len(y))
        s, t = perm[:100], perm[100:]
        fitted = model.fit(X[s], y[s])
        row = {"test": fitted.score(X[t], y[t])}
        for K in Ks:
            row[f"cv{K}"] = cross_val_score(model, X[s], y[s], cv=KFold(K, shuffle=True, random_state=rep)).mean()
        rows.append(row)
    runs = pd.DataFrame(rows)
    table = pd.DataFrame({"K": Ks, "training size": [100 - 100 // K for K in Ks],
                          "bias (CV - test)": [(runs[f"cv{K}"] - runs["test"]).mean() for K in Ks],
                          "sd of (CV - test)": [(runs[f"cv{K}"] - runs["test"]).std() for K in Ks]})
    print(f"mean accuracy of the model fitted on 100 patients: {runs['test'].mean():.4f}")
    print(table.round(4).to_string(index=False))

    lc = {}
    for m in (25, 50, 75, 100, 200):
        accs = []
        for _ in range(100):
            perm = rng.permutation(len(y))
            accs.append(model.fit(X[perm[:m]], y[perm[:m]]).score(X[perm[m:]], y[perm[m:]]))
        lc[m] = np.mean(accs)
    print("learning curve of the logistic regression (mean accuracy by training size):", ", ".join(f"{m}: {v:.4f}" for m, v in lc.items()))


solution_5_7()

# %% [markdown]
# The model fitted on 100 patients has a mean accuracy of 0.9651 on the other 469. The CV estimates are again slightly pessimistic, but the bias is **much smaller** than for the 3-NN on Digits of Section 5.3.2: 1.04 points for $K = 2$, 0.37 for $K = 5$, 0.21 for $K = 10$ and 0.19 for $K = 20$, against 11, 3, 1.4 and 0.8 points. The standard deviation of $\mathrm{CV}_K - $ test accuracy is 0.028, 0.026, 0.025 and 0.024: it hardly depends on $K$, and it is smaller than the 0.042 of Digits (accuracies close to 1 leave less room for variation).
#
# The reason is the **learning curve** (last line of the output). The accuracy of the logistic regression grows from 0.9535 with 50 training patients to 0.9647 with 100: halving the training set costs 1.1 points, which is the bias that we measure for $K = 2$ (1.04). For the larger values of $K$ the training size is 80, 90 and 95 and the curve between 75 and 100 patients is almost flat (0.9615 and 0.9647), which accounts for biases of a few tenths of a point. A linear model with 30 features is already close to its best on 100 patients of this easy problem, while a 10-class nearest-neighbor classifier with 100 images is still far from it. The lesson is that the choice of $K$ should be made by looking at the learning curve of **the model at hand**: for a model that saturates early, $K = 5$ is as good as $K = 20$; for a data-hungry model, larger $K$ is worth its cost.

# %%
def solution_5_8():
    from sklearn.base import clone
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)     # a fresh split
    pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = {"knn__n_neighbors": list(range(1, 31, 2)), "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(X_tr, y_tr)
    cv_scores = np.array(gs.cv_results_["mean_test_score"])
    test_scores = np.array([clone(pipe).set_params(**p).fit(X_tr, y_tr).score(X_te, y_te) for p in gs.cv_results_["params"]])
    print(f"{len(cv_scores)} configurations; CV accuracy ranges from {cv_scores.min():.3f} to {cv_scores.max():.3f}, "
          f"test accuracy from {test_scores.min():.3f} to {test_scores.max():.3f}")

    rng = np.random.default_rng(SEED)
    budgets = [2, 4, 8, 15, 30, 60]
    rows = []
    for B in budgets:
        picked_cv, picked_test = [], []
        for _ in range(500):
            idx = rng.choice(len(cv_scores), size=B, replace=False)
            best = idx[np.argmax(cv_scores[idx])]
            picked_cv.append(cv_scores[best]); picked_test.append(test_scores[best])
        rows.append({"budget B": B, "CV accuracy of the selected": np.mean(picked_cv), "test accuracy of the selected": np.mean(picked_test)})
    table = pd.DataFrame(rows)
    table["optimism"] = table["CV accuracy of the selected"] - table["test accuracy of the selected"]
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(table["budget B"], table["CV accuracy of the selected"], color=VERMILION, marker="o", linewidth=2, label="CV accuracy of the selected configuration")
    ax.plot(table["budget B"], table["test accuracy of the selected"], color=BLUE, marker="s", linestyle="--", linewidth=2, label="test accuracy of the selected configuration")
    ax.set_xscale("log")
    ax.set_xticks(budgets, labels=[str(b) for b in budgets])
    ax.minorticks_off()
    ax.set(xlabel="budget $B$ (configurations evaluated)", ylabel="accuracy", title="Random search on the $k$-NN grid")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    plt.show()


solution_5_8()

# %% [markdown]
# The 60 configurations are all reasonable: their CV accuracies range from 0.948 to 0.972 and their test accuracies from 0.944 to 0.958, and the test set has 143 patients, so one patient is 0.7 points. The table and the figure show what a random search with budget $B$ obtains. Most of the gain comes from the first evaluations: the test accuracy of the selected configuration rises from 0.9526 at $B = 2$ to 0.9554 at $B = 8$ and to 0.9566 at $B = 15$, and then only to 0.9580 at $B = 60$. From $B = 15$ to $B = 60$ the gain is 0.0014, about **one fifth of one patient**, while the cost is four times larger: the search stops paying at a budget of about 8 to 15 configurations, that is, 13% to 25% of the grid. This is the argument of Bergstra and Bengio (2012) in miniature: when many configurations are almost equivalent, a few random draws are almost as good as the full grid.
#
# The difference between the two curves is the optimism of the selection. It is 1.1 points at $B = 2$, 1.4 points from $B = 8$ on, and **0.0139 at $B = 60$**: the CV accuracy of the winner of the whole grid, 0.9719, is 1.4 points higher than the accuracy that the same configuration achieves on new patients. Unlike the coin-flip experiment of Exercise 5.9, the optimism does not keep growing with the number of candidates, because the configurations are not independent: neighboring values of $k$, or the same $k$ with the two distances, give nearly the same predictions and nearly the same scores, so evaluating 60 of them is like evaluating a handful of really different ones. This is why, with real hyperparameter grids, the optimism is moderate; and it is why the final number to report is the one measured on the test set and not the best CV score.

# %%
def solution_5_9():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, _ = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEED)

    def best_cv_optimism(n, ms, m_max, n_rep=40):
        """Mean over repetitions of (best CV accuracy among the first m candidates) - 0.5, for each m in ms."""
        out = {m: [] for m in ms}
        for rep in range(n_rep):
            idx = rng.choice(len(X), size=n, replace=False)
            y_coin = rng.integers(0, 2, size=n)                                # labels unrelated to the features
            scores = []
            for _ in range(m_max):
                cols = rng.choice(X.shape[1], size=5, replace=False)
                model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
                scores.append(cross_val_score(model, X[idx][:, cols], y_coin, cv=KFold(5, shuffle=True, random_state=rep)).mean())
            for m in ms:
                out[m].append(max(scores[:m]) - 0.5)
        return {m: np.mean(v) for m, v in out.items()}

    ms = [1, 4, 16, 64]
    res_m = best_cv_optimism(60, ms, 64)
    print("(a) n = 60")
    print(pd.DataFrame({"m": ms, "mean optimism": [res_m[m] for m in ms], "bound sqrt(ln m/(2n))": [np.sqrt(np.log(m) / 120) for m in ms]}).round(4).to_string(index=False))
    ns = [30, 60, 120, 240]
    res_n = {n: best_cv_optimism(n, [16], 16)[16] for n in ns}
    print("(b) m = 16")
    print(pd.DataFrame({"n": ns, "mean optimism": [res_n[n] for n in ns], "bound sqrt(ln m/(2n))": [np.sqrt(np.log(16) / (2 * n)) for n in ns]}).round(4).to_string(index=False))


solution_5_9()

# %% [markdown]
# **(a)** For $n = 60$ the mean optimism (best CV accuracy minus the true accuracy 0.5) is 0.019 for $m = 1$, 0.062 for $m = 4$, 0.111 for $m = 16$ and 0.139 for $m = 64$. The value for $m = 1$ should be zero, because a single candidate is not selected: 0.019 is the Monte Carlo error of 40 repetitions (the standard deviation of one CV accuracy on 60 coin flips is about $0.5/\sqrt{60} = 0.065$, so the mean of 40 has an error of about 0.01). **(b)** For $m = 16$ the optimism is 0.124, 0.108, 0.091 and 0.065 for $n = 30, 60, 120, 240$; the value 0.108 at $n = 60$ differs from the 0.111 of part (a) for the same reason, since the two runs use different random draws.
#
# **(c)** The bound $\sqrt{\ln m/(2n)}$ is 0, 0.108, 0.152 and 0.186 for the $m$ of part (a), and 0.215, 0.152, 0.108 and 0.076 for the $n$ of part (b): the measured optimism is **below the bound in every row**, as it must be. The qualitative shape is the predicted one: the optimism increases with the number of candidates, with diminishing increments (it grows by 0.049 from $m = 4$ to $m = 16$ but only by 0.028 from $m = 16$ to $m = 64$, as a function of $\ln m$ should), and it decreases when $n$ grows. The quantitative rate is not reproduced exactly: multiplying $n$ by 8 divides the bound by $\sqrt 8 = 2.8$, but the measured optimism only by 1.9, so the ratio of the measurement to the bound goes from 0.58 at $n = 30$ to 0.86 at $n = 240$. Two cautions are in order. The bound is derived for the error on a single validation set of $n$ examples, while here the estimate is a 5-fold CV accuracy on the same $n$ examples, so the comparison is a heuristic one; and it is a worst-case guarantee for any $m$ candidates, not an asymptotic equality, while our candidates share features and are therefore correlated. The practical message is the one of Section 5.2.3: with few examples, tuning a handful of candidates is enough to produce a CV accuracy that is more than **10 points** above the truth on labels that contain no signal at all.

# %%
def solution_5_10():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y                                                 # 1 = malignant
    ts = np.linspace(0.005, 0.995, 199)
    ratios = [1, 5, 10, 50]

    def mean_cost(y_true, p, t, c_fn):
        flagged = p >= t
        return (np.sum(flagged & (y_true == 0)) + c_fn * np.sum(~flagged & (y_true == 1))) / len(y_true)

    records = []
    for split in range(50):
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=split, stratify=y)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        oof = cross_val_predict(model, X1, y1, cv=StratifiedKFold(10, shuffle=True, random_state=split), method="predict_proba")[:, 1]
        p_test = model.fit(X1, y1).predict_proba(X2)[:, 1]
        for c_fn in ratios:
            t_data = ts[np.argmin([mean_cost(y1, oof, t, c_fn) for t in ts])]
            t_star = 1 / (1 + c_fn)
            records.append({"ratio": c_fn, "t_data": t_data, "t_star": t_star,
                            "cost 0.5": mean_cost(y2, p_test, 0.5, c_fn),
                            "cost theory": mean_cost(y2, p_test, t_star, c_fn),
                            "cost out-of-fold": mean_cost(y2, p_test, t_data, c_fn)})
    df = pd.DataFrame(records)
    summary = df.groupby("ratio").agg(**{"t* (theory)": ("t_star", "first"), "mean OOF threshold": ("t_data", "mean"), "sd OOF threshold": ("t_data", "std"),
                                          "cost, t = 0.5": ("cost 0.5", "mean"), "sd (0.5)": ("cost 0.5", "std"),
                                          "cost, t*": ("cost theory", "mean"), "sd (t*)": ("cost theory", "std"),
                                          "cost, OOF": ("cost out-of-fold", "mean"), "sd (OOF)": ("cost out-of-fold", "std")})
    print(summary.round(3).to_string())


solution_5_10()

# %% [markdown]
# The table gives, for each ratio $c_{FN}/c_{FP}$, the theoretical threshold $t^\ast = 1/(1 + c_{FN}/c_{FP})$, the mean and standard deviation of the threshold chosen from the out-of-fold scores, and the mean cost per patient (with its standard deviation over the 50 splits) of the three rules on the test part.
#
# For $c_{FN}/c_{FP} = 1$ the three rules coincide in practice ($t^\ast = 0.5$ is the conventional threshold, and the data-driven rule is 0.003 worse, the price of estimating a threshold that is already known). For a ratio of 5 the theory and the out-of-fold rule are equivalent (0.092 and 0.090) and both are a little better than 0.5 (0.101), a difference smaller than the spread across splits. As the ratio grows the conventional threshold becomes costly and the **theory threshold is the best rule**: at ratio 10 the costs are 0.196 for $t = 0.5$, 0.156 for the out-of-fold optimum and 0.136 for $t^\ast$; at ratio 50 they are 0.954, 0.331 and 0.256, so that keeping $t = 0.5$ costs 3.7 times as much as using $t^\ast$.
#
# The out-of-fold threshold is **highly variable**: its standard deviation is 0.074, 0.110 and 0.126 for the ratios 1, 5 and 10, as large as the threshold itself at ratio 10 (mean 0.162, against $t^\ast = 0.091$). The reason is visible in Figure 5.11(b): the cost curve of the logistic regression is nearly flat over a wide range of thresholds, with steps of one patient, so the minimum of a noisy curve can fall anywhere on the plateau, and the cost on new patients depends on where it fell (the standard deviation of the cost of this rule is 0.070 at ratio 10 and 0.225 at ratio 50, against 0.055 and 0.144 for $t^\ast$). The theory threshold has no such noise, but it is valid only if the scores are calibrated probabilities, which is the case for the logistic regression of this data set (Figure 5.11(a)); for a classifier with poorly calibrated scores, such as a $k$-NN with few neighbors or a decision tree, the threshold would have to be chosen from the data, after calibrating the scores, preferably with a smoother criterion than the minimum of a staircase.

# %% [markdown]
# ## C.6 Chapter 6
#
# ### Manual exercises
#
# **Solution 6.1 — Variances by hand.**
#
# (a) With rows $(1, x_i)$ for $x = -3, -1, 1, 3$: $\mathbf{X}^\top\mathbf{X} = \begin{pmatrix}4 & 0\\ 0 & 20\end{pmatrix}$ (the sum of the $x_i$ is $0$ and the sum of their squares is $9 + 1 + 1 + 9 = 20$). Then $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} = 4\,\mathrm{diag}(\tfrac14, \tfrac1{20}) = \mathrm{diag}(1,\ 0.2)$: $\operatorname{Var}(\hat b) = 1$, $\operatorname{Var}(\hat w) = 0.2$, and the two estimators are uncorrelated because the design is centered.
#
# (b) With $\mathbf{x}_0 = (1, 2)^\top$: $h_0 = \mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0 = \tfrac14 + \tfrac{4}{20} = 0.45$. The variance of the fitted value is $\sigma^2 h_0 = \mathbf{1.8}$, and the expected squared error of a *new observation* is the noise plus that variance: $\sigma^2(1 + h_0) = 4 \cdot 1.45 = \mathbf{5.8}$ (Theorem 6.1).
#
# (c) $h_{ii} = \tfrac14 + x_i^2/20$, that is $0.7,\ 0.3,\ 0.3,\ 0.7$, and the sum is $2 = p$: the trace of the hat matrix equals the number of parameters. The two extreme points have leverage $0.7$, so they pull the line more than the central ones.
#
# (d) For $x = -3, -3, 3, 3$ we get $\sum_i(x_i - \bar x)^2 = 36$ instead of $20$ and $\operatorname{Var}(\hat w) = \sigma^2/36 = 0.111$ instead of $0.2$, a reduction of 44%. The variance of a slope is $\sigma^2/\sum_i(x_i - \bar x)^2$: *spreading the inputs to the ends of the range* is the best way to measure a slope (at the price of not being able to detect curvature, which needs points in the middle).
#
# **Solution 6.2 — Ridge by hand.**
#
# (a) $\mathbf{X}^\top\mathbf{X} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$, $\mathbf{X}^\top\mathbf{y} = (3, 4)^\top$, and $(\mathbf{X}^\top\mathbf{X})^{-1} = \tfrac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, so $\hat{\mathbf{w}}_0 = \tfrac13(6 - 4,\ -3 + 8) = (\tfrac23, \tfrac53)$. The fitted values are $(\tfrac23, \tfrac53, \tfrac73)$, the residuals $(\tfrac13, \tfrac13, -\tfrac13)$ and $\mathrm{RSS} = \tfrac13$.
#
# (b) $\mathbf{X}^\top\mathbf{X} + \mathbf{I} = \begin{pmatrix}3&1\\1&3\end{pmatrix}$, with inverse $\tfrac18\begin{pmatrix}3&-1\\-1&3\end{pmatrix}$, so $\hat{\mathbf{w}}_1 = \tfrac18(9 - 4,\ -3 + 12) = (\tfrac58, \tfrac98)$. The fitted values are $(\tfrac58, \tfrac98, \tfrac{14}8)$, the residuals $(\tfrac38, \tfrac78, \tfrac28)$ and $\mathrm{RSS} = (9 + 49 + 4)/64 = \tfrac{31}{32} = 0.969$: the fit is worse than that of least squares ($\tfrac13$), and the squared norm of the weights has decreased from $\tfrac{29}{9} = 3.22$ to $\tfrac{53}{32} = 1.66$.
#
# (c) For $\tilde{\mathbf{X}} = (\mathbf{X}; \mathbf{I}_2)$ and $\tilde{\mathbf{y}} = (\mathbf{y}; \mathbf{0})$, $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \mathbf{X}^\top\mathbf{X} + \mathbf{I}$ and $\tilde{\mathbf{X}}^\top\tilde{\mathbf{y}} = \mathbf{X}^\top\mathbf{y}$, so the normal equations of the augmented problem are those of ridge with $\lambda = 1$.
#
# (d) $\mathbf{v}_1^\top\hat{\mathbf{w}}_0 = (\tfrac23 + \tfrac53)/\sqrt2 = \tfrac{7}{3\sqrt2}$ and $\mathbf{v}_1^\top\hat{\mathbf{w}}_1 = (\tfrac58 + \tfrac98)/\sqrt2 = \tfrac{7}{4\sqrt2}$: the ratio is $\tfrac34 = \tfrac{3}{3 + 1} = \sigma_1^2/(\sigma_1^2 + \lambda)$. Along $\mathbf{v}_2$: $(\tfrac23 - \tfrac53)/\sqrt2 = -\tfrac1{\sqrt2}$ and $(\tfrac58 - \tfrac98)/\sqrt2 = -\tfrac{1}{2\sqrt2}$, ratio $\tfrac12 = \tfrac{1}{1 + 1}$. The direction with the smaller singular value is shrunk more. Finally $\mathrm{df}(1) = \tfrac34 + \tfrac12 = \tfrac54$: ridge with $\lambda = 1$ has an effective $1.25$ parameters out of 2.
#
# **Solution 6.3 — Logistic regression: a Newton step, and divergence.**
#
# (a) At $(b, w) = (0, 0)$, $\mathbf{p} = (\tfrac12, \tfrac12, \tfrac12)$, so $\mathbf{p} - \mathbf{y} = (\tfrac12, -\tfrac12, -\tfrac12)$. With the rows $(1, -1), (1, 0), (1, 1)$ of $\tilde{\mathbf{X}}$, the gradient is $\tilde{\mathbf{X}}^\top(\mathbf{p} - \mathbf{y}) = (\tfrac12 - \tfrac12 - \tfrac12,\ -\tfrac12 + 0 - \tfrac12) = (-\tfrac12, -1)$. The Hessian is $\tfrac14\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \tfrac14\mathrm{diag}(3, 2) = \mathrm{diag}(\tfrac34, \tfrac12)$. The Newton step is $-\mathbf{H}^{-1}\mathbf{g} = (\tfrac43\cdot\tfrac12,\ 2\cdot 1) = (\tfrac23, 2)$: the model starts to say that larger $x$ means $y = 1$, with a steep slope.
#
# (b) For $x = -1, y = 0$ the signed score is $m = (2\cdot 0 - 1)(w \cdot (-1)) = w$, and for $x = 1, y = 1$ it is $m = w$: so $L(w) = 2\log(1 + e^{-w})$. Then $L'(w) = -2\sigma(-w)$ and $L''(w) = 2\sigma(w)\sigma(-w)$, and Newton's method gives
# $$
# w_{t+1} = w_t - \frac{L'(w_t)}{L''(w_t)} = w_t + \frac{1}{\sigma(w_t)} = w_t + 1 + e^{-w_t}.
# $$
# From $w_0 = 0$: $w_1 = 2$, $w_2 = 2 + 1 + e^{-2} = 3.135$, $w_3 = 4.179$, $w_4 = 5.194$, $w_5 = 6.200$. The sequence grows by slightly more than $1$ at each step: **linearly, without limit**, while the loss $2\log(1 + e^{-w}) \to 0$ like $e^{-t}$. This is Proposition 6.10 in one dimension: the data are separable (any $w > 0$ classifies both points correctly) and there is no maximum-likelihood estimate.
#
# (c) $J'(w) = -\sigma(-w) + \lambda w$ and $J''(w) = \sigma(w)\sigma(-w) + \lambda > 0$, so $J$ is strictly convex and $J'$ is strictly increasing. Since $J'(0) = -\tfrac12 < 0$ and $J'(1/\lambda) = 1 - \sigma(-1/\lambda) > 0$, $J'$ has exactly one root, in $(0, 1/\lambda)$: the minimizer exists and is unique, and it solves $\sigma(-w) = \lambda w$, that is, $1/(1 + e^{w}) = \lambda w$. For $\lambda = 0.1$ bisection gives $w^\ast \approx \mathbf{1.634}$ (check: $\sigma(-1.634) = 0.1634 = 0.1 \cdot 1.634$). The Newton step for the penalized problem is $w \leftarrow w - J'(w)/J''(w)$ and converges in a few iterations. The penalty has turned a problem without a solution into one with a finite answer, here a model that gives probability $\sigma(1.634) = 0.837$ to the correct class of each example, instead of the probability tending to $1$.
#
# **Solution 6.4 — Ridge shrinks the norm, not each coefficient.**
#
# (a) In the SVD basis $\hat{\mathbf{w}}_\lambda = \sum_j\frac{\sigma_j c_j}{\sigma_j^2 + \lambda}\mathbf{v}_j$, and since the $\mathbf{v}_j$ are orthonormal, $\lVert\hat{\mathbf{w}}_\lambda\rVert^2 = \sum_j\frac{\sigma_j^2c_j^2}{(\sigma_j^2 + \lambda)^2}$. Each term with $c_j \ne 0$ is strictly decreasing in $\lambda$, so the norm is strictly decreasing. For the residual, write $\mathbf{y} = \sum_jc_j\mathbf{u}_j + \mathbf{r}$ with $\mathbf{r}$ orthogonal to the columns of $\mathbf{U}$; then $\mathbf{y} - \mathbf{Z}\hat{\mathbf{w}}_\lambda = \sum_j(1 - f_j)c_j\mathbf{u}_j + \mathbf{r}$ with $1 - f_j = \lambda/(\sigma_j^2 + \lambda)$, so
# $$
# \mathrm{RSS}(\lambda) = \sum_j\Big(\frac{\lambda}{\sigma_j^2 + \lambda}\Big)^2c_j^2 + \lVert\mathbf{r}\rVert^2,
# $$
# and every term increases with $\lambda$. Ridge trades a worse fit for a smaller norm, and does so monotonically.
#
# (b) For the data of Exercise 6.2, $\sigma_1^2 = 3$ and $\sigma_2^2 = 1$, and the components of $\hat{\mathbf{w}}_0$ along $\mathbf{v}_1, \mathbf{v}_2$ are $\tfrac{7}{3\sqrt2}$ and $-\tfrac1{\sqrt2}$. Shrinking them by $\tfrac{3}{3 + \lambda}$ and $\tfrac{1}{1 + \lambda}$,
# $$
# w_1(\lambda) = \frac{1}{\sqrt2}\Big(\frac{7}{\sqrt2(3 + \lambda)} - \frac{1}{\sqrt2(1 + \lambda)}\Big) = \frac{7}{2(3 + \lambda)} - \frac{1}{2(1 + \lambda)}, \qquad
# w_2(\lambda) = \frac{7}{2(3 + \lambda)} + \frac{1}{2(1 + \lambda)}.
# $$
# (Checks: $w_1(0) = \tfrac23$, $w_2(0) = \tfrac53$, and $w_1(1) = \tfrac58$, $w_2(1) = \tfrac98$.) Then $w_1'(\lambda) = -\frac{7}{2(3 + \lambda)^2} + \frac{1}{2(1 + \lambda)^2}$ and $w_1'(0) = -\tfrac7{18} + \tfrac12 = \tfrac19 > 0$: the coefficient **increases** for small $\lambda$. It vanishes when $7(1 + \lambda)^2 = (3 + \lambda)^2$, that is $\sqrt7(1 + \lambda) = 3 + \lambda$, $\lambda^\ast = \frac{3 - \sqrt7}{\sqrt7 - 1} = 0.215$, where $w_1 = 0.677 > w_1(0) = 0.667$; after that $w_1$ decreases to $0$. The other coefficient, $w_2$, decreases throughout.
#
# (c) The norm of the whole vector shrinks monotonically, but a *single* coefficient is a combination of several singular directions that are shrunk at different rates, and it can move in either direction (in Figure 6.3 `s2` even changes sign). The path of a coefficient is therefore not a measure of its importance, and the sign of a coefficient at a given $\lambda$ is not a reliable indicator of the direction of the effect, particularly when features are correlated.
#
# **Solution 6.5 — The softmax model.**
#
# (a) For one example of class $y$, with the one-hot vector $y_k$ ($\sum_ky_k = 1$), $L = -\sum_ky_k\log p_k = -\sum_ky_kz_k + \log\sum_je^{z_j}$. Since $\partial\log\sum_je^{z_j}/\partial z_k = p_k$, we get $\partial L/\partial z_k = p_k - y_k$. The chain rule with $z_k = \mathbf{w}_k^\top\mathbf{x} + b_k$ gives $\nabla_{\mathbf{w}_k}L = \sum_i(p_{ik} - y_{ik})\mathbf{x}_i$ and $\partial L/\partial b_k = \sum_i(p_{ik} - y_{ik})$.
#
# (b) For $K = 2$, $p_1 = \dfrac{e^{z_1}}{e^{z_0} + e^{z_1}} = \dfrac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$ with $z_1 - z_0 = (\mathbf{w}_1 - \mathbf{w}_0)^\top\mathbf{x} + (b_1 - b_0)$: a logistic regression with $\mathbf{w} = \mathbf{w}_1 - \mathbf{w}_0$ and $b = b_1 - b_0$. The gradients of (a) are $\nabla_{\mathbf{w}_1}L = \sum_i(p_i - y_i)\mathbf{x}_i$, exactly that of Proposition 6.9 (with $p_i = p_{i1}$, $y_i = y_{i1}$), and $\nabla_{\mathbf{w}_0}L = -\nabla_{\mathbf{w}_1}L$ because $p_{i0} - y_{i0} = -(p_{i1} - y_{i1})$. The two gradients add up to zero, which is the infinitesimal form of the invariance of (c).
#
# (c) With $\mathbf{w}_k + \mathbf{c}$ in place of $\mathbf{w}_k$ every score changes to $z_{k} + \mathbf{c}^\top\mathbf{x}$, and $e^{z_k + \mathbf{c}^\top\mathbf{x}}/\sum_je^{z_j + \mathbf{c}^\top\mathbf{x}} = e^{z_k}/\sum_je^{z_j}$: the probabilities, and hence the loss $L$, do not change. Now let $\bar{\mathbf{w}} = \frac1K\sum_k\mathbf{w}_k$ and replace each $\mathbf{w}_k$ by $\mathbf{w}_k - \bar{\mathbf{w}}$. The loss $L$ is unchanged, and since $\sum_k\lVert\mathbf{w}_k - \bar{\mathbf{w}}\rVert^2 = \sum_k\lVert\mathbf{w}_k\rVert^2 - K\lVert\bar{\mathbf{w}}\rVert^2$, the penalty decreases strictly unless $\bar{\mathbf{w}} = \mathbf{0}$. A minimizer for which $\bar{\mathbf{w}} \ne \mathbf{0}$ could be improved, so every minimizer has $\sum_k\mathbf{w}_k = \mathbf{0}$. The intercepts are not penalized and are not constrained: the same shift would apply to them with no change in the penalty.

# %% [markdown]
# ### Python exercises

# %%
def solution_6_6():
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    X, y = load_diabetes(return_X_y=True)
    n_test = 111
    for n_train in [331, 60]:
        d, ols_mse, ridge_mse, alphas = [], [], [], []
        for r in range(100):
            Xa, Xb, ya, yb = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=r)
            ols = make_pipeline(StandardScaler(), LinearRegression()).fit(Xa, ya)
            ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 61))).fit(Xa, ya)
            e_ols, e_ridge = np.mean((yb - ols.predict(Xb)) ** 2), np.mean((yb - ridge.predict(Xb)) ** 2)
            ols_mse.append(e_ols); ridge_mse.append(e_ridge); d.append(e_ols - e_ridge); alphas.append(ridge[-1].alpha_)
        d = np.array(d)
        t = d.mean() / np.sqrt((1 / len(d) + n_test / n_train) * d.var(ddof=1))
        t_naive = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
        p_corr = 2 * stats.t.sf(abs(t), df=len(d) - 1)
        print(f"n_train = {n_train}, n_test = {n_test}:  mean test MSE  least squares {np.mean(ols_mse):.1f}, ridge {np.mean(ridge_mse):.1f};  "
              f"mean difference (OLS - ridge) {d.mean():.1f};  ridge better in {int((d > 0).sum())}/100;  median alpha {np.median(alphas):.1f}")
        print(f"      naive t = {t_naive:.2f} (p = {2 * stats.t.sf(abs(t_naive), df=len(d) - 1):.4f});  corrected t = {t:.2f} (p = {p_corr:.3f})")


solution_6_6()

# %% [markdown]
# **(a)** With 331 training patients, least squares has a mean test MSE of 3024.0 and ridge 3032.1: ridge is slightly *worse* by 8.1 on average and better in 52 of the 100 splits. The naive paired $t$-test, which treats the 100 splits as independent, gives $t = -2.88$ ($p = 0.005$) and would declare ridge *significantly worse*, a false alarm: the splits share patients and the differences are not independent (Section 5.5.2). **(b)** The corrected resampled test gives $t = -0.49$ ($p = 0.63$): there is no evidence of any difference. This is the picture of Section 6.3.4: with $n = 331$ and $p = 10$ there is little for ridge to gain.
#
# **(c)** With 60 training patients the picture changes: least squares has a test MSE of 3672.3 and ridge 3411.0, a mean gain of 261.3 (7%), and ridge is better in 84 of the 100 splits (the median penalty chosen is $\alpha = 15.8$). Ridge lowers the variance of the estimator, which matters when $n$ is small, as in Section 6.3.4. The naive test is wildly confident ($t = 5.5$) but the *corrected* test is not: $t = 0.40$, $p = 0.69$. With a data set of only 442 patients the splits overlap heavily and the gain varies a lot from split to split, and the correction (which inflates the variance by $1/J + n_{\text{test}}/n_{\text{train}} = 1.86$) is conservative in this setting. The honest conclusion is that ridge was better in most of the splits, and that this data set is too small to turn the observation into a statistically significant statement; a new sample of patients would be needed for that.

# %%
def solution_6_7():
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_diabetes(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=SEED)
    Z = StandardScaler().fit_transform(Xtr)

    def ridge_svd(Z, y, lam):
        xm, ym = Z.mean(axis=0), y.mean()
        U, s, Vt = np.linalg.svd(Z - xm, full_matrices=False)
        w = Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ (y - ym)))
        return ym - xm @ w, w

    def ridge_aug(Z, y, lam):
        xm, ym = Z.mean(axis=0), y.mean()
        d = Z.shape[1]
        A = np.vstack([Z - xm, np.sqrt(lam) * np.eye(d)])
        w = np.linalg.lstsq(A, np.r_[y - ym, np.zeros(d)], rcond=None)[0]
        return ym - xm @ w, w

    def ridge_normal(Z, y, lam):
        xm, ym = Z.mean(axis=0), y.mean()
        Zc = Z - xm
        w = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ (y - ym))
        return ym - xm @ w, w

    ref = Ridge(alpha=10.0).fit(Z, ytr)
    print("(a) maximum difference from scikit-learn's Ridge(alpha=10):")
    for name, f in [("SVD", ridge_svd), ("augmented", ridge_aug), ("normal equations", ridge_normal)]:
        b, w = f(Z, ytr, 10.0)
        print(f"    {name:<17} weights {np.abs(w - ref.coef_).max():.1e}   intercept {abs(b - ref.intercept_):.1e}")

    # (b), (c): df and GCV from one SVD
    lams = np.logspace(-2, 4, 61)
    n = len(ytr)
    U, s, _ = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    uy = U.T @ (ytr - ytr.mean())
    rss_resid = np.sum((ytr - ytr.mean()) ** 2) - np.sum(uy ** 2)       # part of y orthogonal to the columns of U
    df = np.array([np.sum(s ** 2 / (s ** 2 + l)) for l in lams])
    rss = np.array([np.sum(((l / (s ** 2 + l)) * uy) ** 2) + rss_resid for l in lams])
    gcv = n * rss / (n - 1 - df) ** 2                                   # the trace of the whole smoother is 1 + df (the intercept)
    # leave-one-out curve of Proposition 6.8, for comparison
    h = lambda l: 1 / n + (U ** 2) @ (s ** 2 / (s ** 2 + l))
    loo = np.array([np.mean(((ytr - ytr.mean() - U @ ((s ** 2 / (s ** 2 + l)) * uy)) / (1 - h(l))) ** 2) for l in lams])
    print(f"(b) df(lambda): 10 at lambda = 0.01, {np.interp(np.log10(10), np.log10(lams), df):.2f} at 10, {np.interp(np.log10(1000), np.log10(lams), df):.2f} at 1000, {df[-1]:.2f} at 10^4")
    print(f"(c) minimizer of GCV: lambda = {lams[gcv.argmin()]:.2f};  minimizer of LOO: lambda = {lams[loo.argmin()]:.2f};  "
          f"largest relative difference between GCV and LOO on the grid: {np.max(np.abs(gcv / loo - 1)):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    axes[0].semilogx(lams, df, color=BLUE, lw=2)
    axes[0].set(xlabel=r"$\lambda$", ylabel=r"$\mathrm{df}(\lambda)$", title="(b) Effective degrees of freedom")
    axes[1].semilogx(lams, loo, color=VERMILION, lw=2, label="leave-one-out")
    axes[1].semilogx(lams, gcv, color=GREEN, lw=2, ls="--", label="GCV")
    axes[1].set(xlabel=r"$\lambda$", ylabel="estimated prediction error", title="(c) GCV against LOO")
    axes[1].set(ylim=(3080, 3400), xlim=(lams[0], 1e3))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # (d) the badly conditioned design of Section 6.2.3, tiny penalty, no intercept
    x_grid = np.linspace(0, 1, 50)
    V = np.vander(x_grid, 10, increasing=True)
    w_true = np.ones(10)
    y_exact = V @ w_true
    lam_d = 1e-10
    U9, s9, Vt9 = np.linalg.svd(V, full_matrices=False)
    w_ref = Vt9.T @ ((s9 / (s9 ** 2 + lam_d)) * (U9.T @ y_exact))
    w_a = np.linalg.lstsq(np.vstack([V, np.sqrt(lam_d) * np.eye(10)]), np.r_[y_exact, np.zeros(10)], rcond=None)[0]
    w_n = np.linalg.solve(V.T @ V + lam_d * np.eye(10), V.T @ y_exact)
    rel = lambda w: np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)
    print(f"(d) degree 9, lambda = {lam_d:g}, kappa(X) = {np.linalg.cond(V):.1e}: relative difference from the SVD solution: "
          f"augmented {rel(w_a):.1e}, normal equations {rel(w_n):.1e}")


solution_6_7()

# %% [markdown]
# **(a)** The three implementations agree with scikit-learn's `Ridge(alpha=10)` to $10^{-12}$ or better: on a well-conditioned design (the standardized Diabetes data) the choice of algorithm is a matter of cost, not of accuracy. **(b)** The effective degrees of freedom decrease from 10 (at $\lambda = 0.01$) to 8.66 at $\lambda = 10$, 2.09 at $\lambda = 1000$ and 0.31 at $\lambda = 10^4$.
#
# **(c)** GCV, which replaces each individual leverage $h_{ii}$ by their average $(1 + \mathrm{df})/n$, and the exact leave-one-out error give almost the same curve (the largest relative difference on the grid is 0.27%), but their minimizers differ by a factor of 16 ($\lambda = 25.1$ against $1.58$). The reason is the flatness of the curves: between $\lambda = 10^{-2}$ and $10^2$ the estimated error varies by a few units out of 3100, so a perturbation of a fraction of a percent moves the minimum far along the axis. The practical lesson of Section 6.3.4 applies: the value of $\lambda$ within the flat region hardly matters, and one should not read much into the exact location of the minimum.
#
# **(d)** On the degree-9 monomial design (condition number $3.6 \cdot 10^6$), with $\lambda = 10^{-10}$, least squares on the augmented data agrees with the SVD solution to $10^{-11}$, whereas the normal equations differ by $10^{-5}$: forming $\mathbf{X}^\top\mathbf{X}$ squares the condition number, exactly as in Section 6.2.3, and a penalty as small as $10^{-10}$ does not repair it. When $\lambda$ is large enough compared with the smallest squared singular values, the penalty is itself a cure for ill-conditioning, which is another reason for using ridge in practice.

# %%
def solution_6_8():
    from scipy.optimize import minimize, check_grad
    from scipy.special import softmax, logsumexp
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def unpack(theta, K, d):
        return theta[:K * d].reshape(K, d), theta[K * d:]

    def loss_and_grad(theta, Z, y, K, lam):
        n, d = Z.shape
        W, b = unpack(theta, K, d)
        S = Z @ W.T + b
        logp = S - logsumexp(S, axis=1, keepdims=True)
        Y = np.eye(K)[y]
        J = -np.mean(np.sum(Y * logp, axis=1)) + 0.5 * lam * np.sum(W ** 2)
        P = np.exp(logp)
        gW = (P - Y).T @ Z / n + lam * W
        gb = (P - Y).mean(axis=0)
        return J, np.r_[gW.ravel(), gb]

    # (a) gradient check on random data
    rng = np.random.default_rng(SEED)
    Zr, yr, K, lam = rng.normal(size=(40, 5)), rng.integers(0, 3, 40), 3, 0.3
    th = rng.normal(size=3 * 5 + 3)
    err = check_grad(lambda t: loss_and_grad(t, Zr, yr, K, lam)[0], lambda t: loss_and_grad(t, Zr, yr, K, lam)[1], th)
    print(f"(a) norm of (analytic gradient - finite differences) = {err:.1e}   (norm of the gradient = {np.linalg.norm(loss_and_grad(th, Zr, yr, K, lam)[1]):.2f})")

    # (b) Iris
    X, y = load_iris(return_X_y=True)
    Z = StandardScaler().fit_transform(X)
    n, d = Z.shape
    K, lam = 3, 0.01
    res = minimize(loss_and_grad, np.zeros(K * d + K), args=(Z, y, K, lam), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-15})
    W, b = unpack(res.x, K, d)
    sk = LogisticRegression(C=1 / (n * lam), max_iter=10000, tol=1e-12).fit(Z, y)
    print(f"(b) L-BFGS: {res.nit} iterations, J = {res.fun:.6f}, norm of the gradient {np.linalg.norm(res.jac):.1e};  "
          f"max |W - sklearn| = {np.abs(W - sk.coef_).max():.1e},  max |b - sklearn| = {np.abs(b - sk.intercept_).max():.1e}")
    print(f"(c) sum over the classes of the weights, per feature: {np.round(W.sum(axis=0), 8)}")
    print("    the same, for the same model with an arbitrary vector added to all the class weights: probabilities change by "
          f"{np.abs(softmax(Z @ (W + 3.0).T + b, axis=1) - softmax(Z @ W.T + b, axis=1)).max():.1e}")


solution_6_8()

# %% [markdown]
# **(a)** The analytic gradient agrees with finite differences to $1.3 \cdot 10^{-7}$, while the gradient itself has norm 1.63: the implementation is correct. **(b)** L-BFGS converges in 35 iterations to $J = 0.243677$, with a gradient of norm $4 \cdot 10^{-9}$, and the weights agree with those of scikit-learn's `LogisticRegression(C=1/(n*lam))` to $5 \cdot 10^{-7}$ (the intercepts to $9 \cdot 10^{-7}$), which confirms the correspondence $C = 1/(n\lambda)$ of Section 6.5 for the multinomial model. **(c)** The sum over the three classes of the weights of each feature is zero to eight decimals, as shown in Exercise 6.5, and adding the same constant to all the class weights changes the probabilities only at the level of rounding, $6 \cdot 10^{-16}$.

# %%
def solution_6_9():
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler

    diab = load_diabetes(as_frame=True)
    y = diab.target.to_numpy()
    Z0 = StandardScaler().fit_transform(diab.data)
    names = list(diab.data.columns)
    j = names.index("bmi")
    n = len(y)
    eps = np.random.default_rng(SEED).normal(size=n)
    B_boot, lam_r = 300, 10.0

    def fit_ols(Z, y):
        Zc = Z - Z.mean(axis=0)
        return np.linalg.lstsq(Zc, y - y.mean(), rcond=None)[0]

    def fit_ridge(Z, y, lam):
        Zc = Z - Z.mean(axis=0)
        return np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ (y - y.mean()))

    rows = []
    for rho in [0.0, 0.9, 0.99, 0.999]:
        col = rho * Z0[:, j] + np.sqrt(1 - rho ** 2) * eps
        col = (col - col.mean()) / col.std()
        Z = np.column_stack([Z0, col])                                    # the last column is the constructed copy
        Zc = Z - Z.mean(axis=0)
        w = fit_ols(Z, y)
        sigma2 = np.sum((y - y.mean() - Zc @ w) ** 2) / (n - Z.shape[1] - 1)
        se = np.sqrt(sigma2 * np.linalg.inv(Zc.T @ Zc)[j, j])
        vif = np.linalg.inv(np.corrcoef(Z, rowvar=False))[j, j]
        rng = np.random.default_rng(SEED)
        bo, so, br, sr = [], [], [], []
        for _ in range(B_boot):
            idx = rng.integers(0, n, n)
            wo, wr = fit_ols(Z[idx], y[idx]), fit_ridge(Z[idx], y[idx], lam_r)
            bo.append(wo[j]); so.append(wo[j] + wo[-1]); br.append(wr[j]); sr.append(wr[j] + wr[-1])
        rows.append({"rho": rho, "VIF of bmi": vif, "SE formula": se, "sigma*sqrt(VIF/n)": np.sqrt(sigma2 * vif / n),
                     "OLS: SD bmi": np.std(bo, ddof=1), "OLS: SD of sum": np.std(so, ddof=1),
                     "ridge: SD bmi": np.std(br, ddof=1), "ridge: SD of sum": np.std(sr, ddof=1)})
    print(pd.DataFrame(rows).set_index("rho").to_string(float_format=lambda v: f"{v:.2f}"))


solution_6_9()

# %% [markdown]
# **(a)** For standardized columns $[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj} = \mathrm{VIF}_j/n$, so the standard error is exactly $\hat\sigma\sqrt{\mathrm{VIF}_j/n}$ (the two columns of the table agree). As the constructed copy becomes more similar to `bmi`, the VIF goes from 1.5 ($\rho = 0$) to 6.3, 56.7 and 559.2, and the standard error of the coefficient of `bmi` from 3.17 to 6.50, 19.42 and 60.96: a factor of 19.2, equal to $\sqrt{559.2/1.51}$.
#
# **(b)** The bootstrap standard deviations, 3.29, 6.18, 18.42 and 58.10, agree with the formula to within 10%: the theory holds. **(c)** The standard deviation of the **sum** of the two coefficients stays at about 3.3 for $\rho \ge 0.9$, however collinear the columns are. The data determine very well the *combined* effect of the two nearly identical features and cannot say how it is divided between them: the individual coefficients are unreliable, the prediction is not. (For $\rho = 0$ the sum has a larger deviation, 4.2, because the second column is then pure noise, which adds variance without information.)
#
# **(d)** Ridge ($\lambda = 10$) cuts the standard deviation of each coefficient (4.95 against 6.18 at $\rho = 0.9$, 5.29 against 18.42 at $\rho = 0.99$, 2.63 against 58.10 at $\rho = 0.999$) while the standard deviation of the sum stays at about 3.2: the penalty shares the weight between the two copies, which is the stable thing to do, and makes the individual coefficients readable again (at the price of a bias that the table does not show).

# %%
def solution_6_10():
    import warnings
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    C_grid = 10.0 ** np.arange(-3, 3.01, 0.5)
    clip = lambda p: np.clip(p, 1e-15, 1 - 1e-15)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in [30, 60, 120, 240]:
            rng = np.random.default_rng(SEED)
            rec = {"C_cv": [], "C_oracle": [], "ll_cv": [], "ll_oracle": [], "ll_unpen": []}
            kept = 0
            while kept < 30:
                perm = rng.permutation(len(y))
                tr, te = perm[:n], perm[n:]
                if min(y[tr].sum(), n - y[tr].sum()) < 6:
                    continue
                kept += 1
                cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(10 ** 6)))
                cv_ll, te_ll = [], []
                for C in C_grid:
                    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                    p_cv = cross_val_predict(pipe, X[tr], y[tr], cv=cv, method="predict_proba")[:, 1]
                    cv_ll.append(log_loss(y[tr], clip(p_cv)))
                    te_ll.append(log_loss(y[te], clip(pipe.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])))
                i_cv, i_or = int(np.argmin(cv_ll)), int(np.argmin(te_ll))
                unpen = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000)).fit(X[tr], y[tr])
                rec["C_cv"].append(C_grid[i_cv]); rec["C_oracle"].append(C_grid[i_or])
                rec["ll_cv"].append(te_ll[i_cv]); rec["ll_oracle"].append(te_ll[i_or])
                rec["ll_unpen"].append(log_loss(y[te], clip(unpen.predict_proba(X[te])[:, 1])))
            rows.append({"n": n, "median C (CV)": np.median(rec["C_cv"]), "median C (oracle)": np.median(rec["C_oracle"]),
                         "lambda (CV)": 1 / (n * np.median(rec["C_cv"])), "lambda (oracle)": 1 / (n * np.median(rec["C_oracle"])),
                         "test log-loss, CV choice": np.mean(rec["ll_cv"]), "oracle": np.mean(rec["ll_oracle"]), "unpenalized": np.mean(rec["ll_unpen"])})
    print(pd.DataFrame(rows).set_index("n").to_string(float_format=lambda v: f"{v:.4g}"))


solution_6_10()

# %% [markdown]
# The best $C$ is remarkably stable. The oracle value (the one that minimizes the test log-loss) is $C = 1$ for every $n$, and so is the median value chosen by cross-validation for $n \ge 60$; only for $n = 30$ is the CV choice erratic (median 20.8): with 30 patients the cross-validated log-loss is a noisy criterion, the winner's-curse effect of Chapter 5. Since $\lambda = 1/(Cn)$, a constant $C$ means that **the best penalty decreases like $1/n$**: $0.0167$, $0.0083$ and $0.0042$ for $n = 60, 120, 240$. In the formulation of Section 6.5, with the loss averaged over the examples, the right strength of the penalty falls as the training set grows, while in the formulation of scikit-learn ($C$ multiplies the *sum* of the losses) it stays the same. This is the quantitative form of the principle that more data can support a more flexible model.
#
# The test log-loss of the model selected by cross-validation decreases from 0.214 to 0.094, and the gap to the oracle shrinks from 0.07 to 0.006 as $n$ grows. The unpenalized model is much worse and does not improve with $n$ (0.387, 0.387, 0.483 and 0.846): with 30 standardized features the training sets are separable, and the solver stops at the iteration limit with large weights and overconfident probabilities (Section 6.4.4); its figures reflect the optimizer's stopping rule, not a property of the model.

# %% [markdown]
# ## C.7 Chapter 7
#
# ### Manual exercises
#
# **Solution 7.1 — Gradient descent by hand.**
#
# (a) $\nabla f = (2w_1, 8w_2)$ and $\mathbf{H} = \mathrm{diag}(2, 8)$, so $\mu = 2$, $L = 8$, $\kappa = 4$, and the minimizer is $\mathbf{w}^\ast = \mathbf{0}$.
#
# (b) With $\eta = 1/L = \tfrac18$ the factors are $1 - \tfrac18\cdot 2 = \tfrac34$ and $1 - \tfrac18\cdot 8 = 0$, so $\mathbf{w}_k = (4\cdot(\tfrac34)^k,\ 0)$ for $k \ge 1$: the iterates are $(3, 0)$, $(2.25, 0)$, $(1.6875, 0)$. The steepest direction is solved in one step (its factor is zero) and the flat one decays like $\tfrac34$ per step, the rate $1 - 1/\kappa$ of Theorem 7.3.
#
# (c) The best step is $\eta^\ast = 2/(L + \mu) = 0.2$, with factors $1 - 0.4 = 0.6$ and $1 - 1.6 = -0.6$: the rate is $\rho = 0.6 = (\kappa - 1)/(\kappa + 1) = \tfrac35$, and both coordinates are multiplied by $\pm 0.6$. The iterates are $(2.4, -0.6)$, $(1.44, 0.36)$, $(0.864, -0.216)$. The step $\eta^\ast$ balances the two directions: the flat one converges from one side and the steep one overshoots to the other, by the same factor.
#
# (d) The iteration converges for $0 < \eta < 2/L = \tfrac14$. For $\eta = \tfrac14$ the factors are $\tfrac12$ and $-1$: $\mathbf{w}_k = (4\cdot 0.5^k,\ (-1)^k)$, which gives $(2, -1)$, $(1, 1)$, $(0.5, -1)$: the first coordinate converges, the second oscillates forever between $\pm 1$, so the iteration does not converge (it is the boundary case of Theorem 7.3, item 1).
#
# **Solution 7.2 — The smoothness constant of a logistic regression.**
#
# (a) With $s = \sigma(m)$, $\ell'(m) = -\sigma(-m)$ and $\ell''(m) = \sigma(m)\sigma(-m) = s(1 - s)$. Since $s \in (0, 1)$, $s(1 - s) \le \tfrac14$ (the maximum, at $s = \tfrac12$, i.e. $m = 0$), and it is positive. The Hessian of the objective is $\frac1n\tilde{\mathbf{X}}^\top\mathrm{diag}(\ell''(m_i))\tilde{\mathbf{X}} + \lambda\,\mathrm{diag}(\text{penalized coordinates}) \preceq \frac1{4n}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} + \lambda\mathbf{I}$ (the signs $2y_i - 1 = \pm 1$ disappear in the squares), whose largest eigenvalue is $\lVert\tilde{\mathbf{X}}\rVert_2^2/(4n) + \lambda$.
#
# (b) For the rows $(1, -1), (1, 0), (1, 1)$ we get $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \mathrm{diag}(3, 2)$, so $\lVert\tilde{\mathbf{X}}\rVert_2^2 = 3$ and $L = 3/(4\cdot 3) = \tfrac14$. At the origin all the scores vanish, $\ell'' = \tfrac14$ for every example, and the Hessian is exactly $\frac1{12}\,\mathrm{diag}(3, 2) = \mathrm{diag}(\tfrac14, \tfrac16)$: its largest eigenvalue is $\tfrac14 = L$. The bound is tight at the origin and is loose elsewhere (Section 7.3.4: fourteen times too large at the minimizer of the Breast Cancer problem).
#
# (c) At the origin $\mathbf{p} - \mathbf{y} = (\tfrac12, -\tfrac12, -\tfrac12)$ and $\nabla f = \frac13\tilde{\mathbf{X}}^\top(\mathbf{p} - \mathbf{y}) = \frac13(-\tfrac12,\ -1) = (-\tfrac16, -\tfrac13)$. With $\eta = 1/L = 4$, $\mathbf{w}_1 = (0, 0) + 4(\tfrac16, \tfrac13) = (\tfrac23, \tfrac43)$. The loss at the origin is $\log 2 = 0.6931$. At $\mathbf{w}_1$ the scores are $-\tfrac23, \tfrac23, 2$ and the mean loss is $\frac13[\log(1 + e^{-2/3}) + \log(1 + e^{-2/3}) + \log(1 + e^{-2})] = \frac13(0.4144 + 0.4144 + 0.1269) = 0.3186$. The descent lemma with $\eta = 1/L$ guarantees $f(\mathbf{w}_1) \le f(\mathbf{w}_0) - \frac1{2L}\lVert\nabla f(\mathbf{w}_0)\rVert^2 = 0.6931 - 2\cdot(\tfrac1{36} + \tfrac19) = 0.6931 - 0.2778 = 0.4153$. The actual value $0.3186$ satisfies the inequality, with a decrease of $0.3745$ against the guaranteed $0.2778$.
#
# **Solution 7.3 — The heavy ball on a quadratic.**
#
# (a) $\kappa = 100$. The best step is $2/(L + \mu) = 2/101$ with $\rho = 99/101 = 0.9802$, and $\rho^k = 10^{-6}$ gives $k = \ln 10^6/\ln(101/99) = 690.8$, so about $691$ iterations (equal to $\frac\kappa2\ln 10^6 = 50\cdot 13.82$, the bound of Theorem 7.3, item 4).
#
# (b) $\sqrt L + \sqrt\mu = 11$, so $\eta = 4/121 = 0.0331$, $\beta = (9/11)^2 = 0.6694$ and $\rho = \sqrt\beta = 9/11 = 0.8182$. Then $k = \ln 10^6/\ln(11/9) = 68.8$: about $69$ iterations, ten times fewer, the factor $\sqrt\kappa = 10$ of the theory.
#
# (c) For $\lambda = \mu = 1$: $1 + \beta - \eta\mu = 1 + 0.6694 - 0.0331 = 1.6364 = 2\cdot\tfrac9{11}$, so $z^2 - \tfrac{18}{11}z + (\tfrac9{11})^2 = (z - \tfrac9{11})^2$: a double root $\tfrac9{11} = \sqrt\beta$. For $\lambda = L = 100$: $1 + \beta - 100\eta = 1.6694 - 3.3058 = -1.6364$, so $z^2 + \tfrac{18}{11}z + (\tfrac9{11})^2 = (z + \tfrac9{11})^2$: a double root $-\tfrac9{11}$, of modulus $\sqrt\beta$. For the eigenvalues between the two the roots are complex conjugates of modulus exactly $\sqrt\beta$ (the product of the roots is $\beta$), so every direction contracts at the same rate $\sqrt\beta$ and the double roots at the two ends produce the factor $(k + 1)$ of Theorem 7.8.
#
# **Solution 7.4 — SGD for a mean.**
#
# (a) The update is $w_{k+1} = w_k - \eta_k(w_k - x_k) = (1 - \eta_k)w_k + \eta_kx_k$. With $\eta_k = 1/(k + 1)$: $\eta_0 = 1$ gives $w_1 = x_0$, and then $(k + 1)w_{k+1} = kw_k + x_k$, so $kw_k = x_0 + \dots + x_{k-1}$: $w_k$ is the sample mean of the first $k$ observations. Its variance is $\sigma^2/k$, and since $f(w) - f^\ast = \tfrac12(w - m)^2$, $\mathbb{E}f(w_k) - f^\ast = \sigma^2/(2k)$. This is the $O(1/k)$ rate of Theorem 7.12, and it is attained exactly: no step schedule can do better than the statistical error of the sample mean. The schedule $\eta_k = 1/(k + 1)$ has $\beta = 1/\mu$ and $\gamma = 1$, the limiting case of the hypothesis $\beta > 1/\mu$ of the theorem, for which its constant $\beta^2L\sigma^2/(2(\beta\mu - 1))$ is infinite; with $\beta = \gamma = 2$ ($\eta_k = 2/(k + 2)$) the theorem gives $\nu = \max\{2\sigma^2, 2(f(w_0) - f^\ast)\}$, of the same order $\sigma^2/k$ but with a larger constant.
#
# (b) Write $w_{k+1} - m = (1 - \eta)(w_k - m) + \eta(x_k - m)$. The two terms are independent, so $V_{k+1} = (1 - \eta)^2V_k + \eta^2\sigma^2$, which converges for $\lvert 1 - \eta\rvert < 1$ to $V_\infty = \eta^2\sigma^2/(1 - (1 - \eta)^2) = \eta\sigma^2/(2 - \eta)$. Then $\mathbb{E}f - f^\ast = \tfrac12V_\infty = \eta\sigma^2/(2(2 - \eta))$, against the floor $L\eta\sigma^2/(2\mu) = \eta\sigma^2/2$ of Theorem 7.11: the floor overestimates the true one by the factor $2 - \eta \in [1, 2)$.
#
# (c) For $\eta = 0.1$, $\sigma^2 = 1$: exact $0.1/(2\cdot 1.9) = 0.02632$, floor $0.05$, ratio $1.9$. The theorem is within a factor of two of the truth in this case, and the exact floor is proportional to $\eta$, as the theorem says.
#
# **Solution 7.5 — Early stopping and ridge in two directions.**
#
# (a) $\varphi_1(k) = 1 - (1 - 1)^k = 1$ for every $k \ge 1$ and $\varphi_2(k) = 1 - 0.99^k$.
#
# (b) The ridge factor with $\lambda = 1/(\eta k) = 1/k$ is $h_j/(h_j + 1/k)$:
#
# | $k$ | $\varphi_2(k) = 1 - 0.99^k$ | ridge $0.01/(0.01 + 1/k)$ | ratio |
# |---|---|---|---|
# | 1 | 0.0100 | 0.0099 | 1.010 |
# | 10 | 0.0956 | 0.0909 | 1.052 |
# | 100 | 0.6340 | 0.5000 | 1.268 |
# | 1000 | 0.99996 | 0.9091 | 1.100 |
#
# (c) We need $1 - 0.99^k \ge \tfrac12$, that is $k \ge \ln 2/(-\ln 0.99) = 68.97$, so $k = 69$. The ridge factor is $\tfrac12$ exactly when $\lambda = h_2 = 0.01$, which corresponds to $k = 1/(\eta\lambda) = 100$ in the correspondence $\lambda = 1/(\eta k)$: gradient descent recovers the second direction half way a little *earlier* than the correspondence predicts, as the ratio $1.27 > 1$ at $k = 100$ shows.
#
# (d) All the ratios of (b) lie between $1$ and $2$. For the first direction the ridge factor is $k/(k + 1)$ and the ratio is $(k + 1)/k$: $2$ at $k = 1$ (the upper bound of Proposition 7.13 is attained), $1.1$ at $k = 10$, and tends to $1$. For the second direction the ratio never exceeds $1.27$ in the table. Both factors increase from $0$ to $1$ together, which is the content of the proposition: stopping at $k$ steps and ridge with $\lambda = 1/(\eta k)$ filter the same directions to within a factor of two.
#
# ### Python exercises

# %%
def solution_7_6():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    diab = load_diabetes()
    Xtr, _, ytr, _ = train_test_split(diab.data, diab.target, test_size=0.25, random_state=SEED)
    Z = StandardScaler().fit_transform(Xtr)
    n = len(ytr)
    yc = ytr - ytr.mean()

    def objective(w, lam):
        r = Z @ w - yc
        return 0.5 * np.mean(r**2) + 0.5 * lam * w @ w, Z.T @ r / n + lam * w

    # (a) gradient check by central differences
    rng = np.random.default_rng(SEED)
    w_try = rng.normal(size=Z.shape[1])
    g_fd = np.array([(objective(w_try + 1e-6 * e, 0.1)[0] - objective(w_try - 1e-6 * e, 0.1)[0]) / 2e-6 for e in np.eye(Z.shape[1])])
    g_an = objective(w_try, 0.1)[1]
    print(f"(a) relative error of the gradient: {np.linalg.norm(g_an - g_fd) / np.linalg.norm(g_fd):.1e}")

    rows = []
    for lam in [0.0, 0.01, 0.1, 1.0]:
        H = Z.T @ Z / n + lam * np.eye(Z.shape[1])
        ev = np.linalg.eigvalsh(H)
        L, mu = ev[-1], ev[0]
        w_star = np.linalg.solve(H, Z.T @ yc / n)
        w = np.zeros(Z.shape[1])
        k = 0
        while np.linalg.norm(w - w_star) > 1e-6 * np.linalg.norm(w_star) and k < 200_000:
            w = w - objective(w, lam)[1] / L
            k += 1
        rows.append({"lambda": lam, "L": L, "mu": mu, "kappa": L / mu, "iterations": k, "kappa ln(1e6)": L / mu * np.log(1e6)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    slope, intercept = np.polyfit(np.log(df["kappa"]), np.log(df["iterations"]), 1)
    print(f"(d) slope of log(iterations) against log(kappa): {slope:.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.loglog(df["kappa"], df["iterations"], "o-", color=BLUE, label="measured")
    ax.loglog(df["kappa"], df["kappa ln(1e6)"], "s--", color=VERMILION, label=r"$\kappa\ln 10^6$")
    ax.set(xlabel=r"condition number $\kappa$", ylabel="iterations", title="Gradient descent on Diabetes, $\\eta = 1/L$")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


solution_7_6()

# %% [markdown]
# The iteration counts follow the condition number almost proportionally: $5{,}677$, $2{,}676$, $426$ and $44$ iterations for $\kappa = 417.8$, $202.7$, $36.7$ and $4.9$, against $\kappa\ln 10^6 = 5{,}773$, $2{,}801$, $508$ and $67$. The bound of Theorem 7.3 is nearly attained for $\lambda = 0$ and $0.01$ ($98\%$ and $96\%$) and loose by about a third for $\lambda = 1$: the bound is the worst case, in which the starting error lies entirely along the flattest direction, and the measured count is smaller when the weight of $\mathbf{w}^\ast$ on that direction is only a part of $\lVert\mathbf{w}^\ast\rVert$. Because the ratio of measured count to bound is not constant, the slope of the log-log fit is $1.09$ and not exactly $1$. The gradient check gives a relative error of $7 \cdot 10^{-9}$. A ridge penalty of $\lambda = 1$ makes the problem $85$ times better conditioned than least squares and the descent $130$ times faster, the practical face of the remark of Section 7.2.2 that regularization helps the optimization as well as the statistics.



# %%
def solution_7_7():
    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    mask = iris.target > 0                                     # versicolor (0) against virginica (1)
    X, y = iris.data[mask], (iris.target[mask] == 2).astype(float)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    Z = StandardScaler().fit_transform(Xtr)
    A = np.c_[np.ones(len(ytr)), Z]
    n, p = A.shape
    lam = 0.01
    pen = np.r_[0.0, np.ones(p - 1)]

    def obj(w):
        s = A @ w
        return np.mean(np.logaddexp(0, s) - ytr * s) + 0.5 * lam * np.sum(pen * w**2), A.T @ (expit(s) - ytr) / n + lam * pen * w

    def hess(w):
        q = expit(A @ w)
        return (A.T * (q * (1 - q))) @ A / n + lam * np.diag(pen)

    w = np.zeros(p)
    for _ in range(30):
        w = w - np.linalg.solve(hess(w), obj(w)[1])
    w_star, f_star = w, obj(w)[0]
    L_glob = np.linalg.eigvalsh(A.T @ A / n)[-1] / 4 + lam
    ev = np.linalg.eigvalsh(hess(w_star))
    L_loc, mu_loc = ev[-1], ev[0]
    print(f"(a) f* = {f_star:.8f}; L_glob = {L_glob:.4f}; local L = {L_loc:.4f}, mu = {mu_loc:.4f}; L_glob/L_loc = {L_glob / L_loc:.1f}")

    def gd(eta, iters=4000):
        w, gaps = np.zeros(p), []
        for _ in range(iters):
            f, g = obj(w)
            gaps.append(f - f_star)
            w = w - eta * g
        return np.array(gaps)

    def first_below(gaps, tol=1e-8):
        above = np.flatnonzero(gaps >= tol)
        k = int(above[-1]) + 1 if len(above) else 0
        return k if k < len(gaps) else None

    curves = {"GD, 1/L_glob": gd(1 / L_glob)}
    best_eta, best_k = None, None
    for eta in np.linspace(1 / L_glob, 4 / L_loc, 60):
        k = first_below(gd(eta))
        if k is not None and (best_k is None or k < best_k):
            best_eta, best_k = eta, k
    curves[f"GD, eta = {best_eta:.2f}"] = gd(best_eta)
    # backtracking (Armijo, halving; the step is multiplied by 2 at the start of each iteration)
    w, eta, gaps_bt, n_grad = np.zeros(p), 1.0, [], 0
    for _ in range(4000):
        f, g = obj(w)
        gaps_bt.append(f - f_star)
        if gaps_bt[-1] < 1e-14:
            break
        eta *= 2
        while obj(w - eta * g)[0] > f - 0.5 * eta * g @ g:
            eta /= 2
        w = w - eta * g
    curves["backtracking"] = np.array(gaps_bt)
    # L-BFGS
    hist, count = [], [0]
    def fun(v):
        count[0] += 1
        return obj(v)
    minimize(fun, np.zeros(p), jac=True, method="L-BFGS-B", callback=lambda v: hist.append((obj(v)[0] - f_star, count[0])),
             options={"maxiter": 500, "gtol": 1e-12, "ftol": 1e-16})
    hist = np.array(hist)
    k_lb = int(np.flatnonzero(hist[:, 0] < 1e-8)[0])
    curves["L-BFGS"] = hist[:, 0]
    # Newton
    w, gaps_nt = np.zeros(p), []
    for _ in range(10):
        f, g = obj(w)
        gaps_nt.append(f - f_star)
        w = w - np.linalg.solve(hess(w), g)
    curves["Newton"] = np.array(gaps_nt)

    print(f"(b)-(c) gradient evaluations to a gap below 1e-8:")
    print(f"   GD 1/L_glob: {first_below(curves['GD, 1/L_glob'])};  GD best constant ({best_eta:.2f}): {best_k};  backtracking: {first_below(curves['backtracking'])};  "
          f"L-BFGS: {int(hist[k_lb, 1])} ({k_lb + 1} iterations);  Newton: {int(np.flatnonzero(np.array(gaps_nt) < 1e-8)[0])} steps")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (name, c), color, ls in zip(curves.items(), [BLUE, GREEN, ORANGE, PURPLE, VERMILION], ["-", "-", "-", "--", "--"]):
        ax.plot(np.maximum(c, 1e-16), color=color, ls=ls, lw=2, label=name)
    ax.set(yscale="log", xlim=(0, 400), ylim=(1e-12, 1), xlabel="iteration", ylabel=r"suboptimality $f - f^\ast$",
           title="Iris, versicolor against virginica")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    plt.show()


solution_7_7()

# %% [markdown]
# On this problem, versicolor against virginica, the two classes overlap (an unpenalized near-hard-margin classifier misclassifies two of the $75$ training flowers and the unpenalized logistic fit has a finite norm), so the penalty $\lambda = 0.01$ is not needed for existence, only for conditioning. The minimum is $f^\ast = 0.15264447$; the global bound is $L_{\text{glob}} = 0.757$, the curvature at the minimizer is $L_{\text{loc}} = 0.0625$ (with $\mu_{\text{loc}} = 0.0136$), a ratio of $12.1$. The numbers of gradient evaluations to reach a suboptimality of $10^{-8}$ are: gradient descent with $\eta = 1/L_{\text{glob}}$, $217$; with the best constant step found in the search, $\eta = 25.8$ (the stability limit is $2/L_{\text{loc}} = 32$), $15$; backtracking, $11$; L-BFGS, $10$ evaluations ($9$ iterations); Newton, $6$ steps. The gain of the best constant step, $217/15 = 14.5$, is of the order of the ratio $L_{\text{glob}}/L_{\text{loc}} = 12.1$, as in the Breast Cancer experiment of Section 7.3.4 (where the best constant step gained $1631/65 = 25$ against a ratio of $14.4$): the loss of a method that obeys the global bound is the ratio between the worst and the local curvature. The curves of the three first-order methods are straight lines on the logarithmic scale (linear convergence), those of L-BFGS and Newton bend downward (superlinear and quadratic).



# %%
def solution_7_8():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    diab = load_diabetes()
    Xtr, _, ytr, _ = train_test_split(diab.data, diab.target, test_size=0.25, random_state=SEED)
    Z = StandardScaler().fit_transform(Xtr)
    n, lam = len(ytr), 0.1
    yc = ytr - ytr.mean()
    H = Z.T @ Z / n + lam * np.eye(Z.shape[1])
    b = Z.T @ yc / n
    ev = np.linalg.eigvalsh(H)
    L, mu = ev[-1], ev[0]
    kappa = L / mu
    w_star = np.linalg.solve(H, b)
    sk = np.sqrt(kappa)

    def run(eta, beta, kind, iters=6000):
        w, w_prev, dist = np.zeros(10), np.zeros(10), np.empty(iters)
        for k in range(iters):
            dist[k] = np.linalg.norm(w - w_star) / np.linalg.norm(w_star)
            if not np.isfinite(dist[k]) or dist[k] > 1e12:
                dist[k:] = np.inf
                break
            if kind == "hb":
                w_new = w - eta * (H @ w - b) + beta * (w - w_prev)
            else:
                y = w + beta * (w - w_prev)
                w_new = y - eta * (H @ y - b)
            w_prev, w = w, w_new
        return dist

    def count(d, tol=1e-8):
        above = np.flatnonzero(d >= tol)
        k = int(above[-1]) + 1 if len(above) else 0
        return k if k < len(d) else None

    eta_hb = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta_hb = ((sk - 1) / (sk + 1)) ** 2
    runs = {
        "GD, eta = 2/(L+mu)": run(2 / (L + mu), 0.0, "hb"),
        "heavy ball (Theorem 7.8)": run(eta_hb, beta_hb, "hb"),
        "Nesterov, strongly convex": run(1 / L, (sk - 1) / (sk + 1), "nag"),
        "heavy ball, beta = 0.99": run(eta_hb, 0.99, "hb"),
        "heavy ball, beta = 0": run(eta_hb, 0.0, "hb"),
    }
    print(f"kappa = {kappa:.2f}, sqrt(kappa) = {sk:.2f}; heavy-ball eta = {eta_hb:.3f} (2/L = {2 / L:.3f}), beta = {beta_hb:.4f}")
    cnt = {k: count(v) for k, v in runs.items()}
    for k, v in cnt.items():
        print(f"  {k:<28} {v if v else 'not reached'}")
    print(f"(a) iterations(GD) / iterations(heavy ball) = {cnt['GD, eta = 2/(L+mu)'] / cnt['heavy ball (Theorem 7.8)']:.1f}, "
          f"iterations(GD) / iterations(Nesterov) = {cnt['GD, eta = 2/(L+mu)'] / cnt['Nesterov, strongly convex']:.1f}; sqrt(kappa) = {sk:.1f}")
    d99 = runs["heavy ball, beta = 0.99"]
    print(f"(b) beta = 0.99: the distance after 100, 1000, 3000 iterations is {d99[100]:.1e}, {d99[1000]:.1e}, {d99[3000]:.1e}; "
          f"it increases at {100 * np.mean(np.diff(d99[:600]) > 0):.0f}% of the first 600 iterations (oscillation)")
    d0 = runs["heavy ball, beta = 0"]
    print(f"    beta = 0 keeps the heavy-ball step {eta_hb:.3f} > 2/L = {2 / L:.3f}: plain gradient descent diverges (distance above 1e12 at iteration {int(np.flatnonzero(~np.isfinite(d0))[0])})")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for (name, d), color, ls in zip(runs.items(), [BLUE, VERMILION, GREEN, ORANGE, PURPLE], ["-", "-", "-", "--", ":"]):
        ax.plot(np.maximum(d, 1e-14), color=color, ls=ls, lw=1.8, label=name)
    ax.set(yscale="log", xlim=(0, 1200), ylim=(1e-12, 1e3), xlabel="iteration $k$", ylabel=r"$\|\mathbf{w}_k - \mathbf{w}^\ast\| / \|\mathbf{w}^\ast\|$",
           title="Diabetes ridge, $\\lambda = 0.1$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    plt.show()


solution_7_8()

# %% [markdown]
# With $\lambda = 0.1$ ($\kappa = 36.74$, $\sqrt\kappa = 6.06$) the counts to reach a relative distance of $10^{-8}$ are $328$ iterations for gradient descent with $\eta = 2/(L + \mu)$ (the bound $\frac\kappa2\ln 10^8 = 338$), $68$ for the heavy ball with the parameters of Theorem 7.8 ($\eta = 0.733$, $\beta = 0.514$), and $107$ for Nesterov's strongly convex version. (a) The ratios are $4.8$ and $3.1$, against $\sqrt\kappa = 6.1$: the heavy ball is not quite $\sqrt\kappa$ times faster at this moderate $\kappa$ because the error bound of Theorem 7.8 carries the factor $(k + 1)$ of the double roots (with only $\rho^k$ the ratio would be $6.1$), and Nesterov's version uses the smaller step $1/L$ and is slower still, as in Section 7.5. (b) The momentum parameter must be matched to the step. With $\beta = 0.99$ the rate falls to $\sqrt{0.99} = 0.995$, and the run needs $3{,}723$ iterations, eleven times more than plain gradient descent: the iterate overshoots and spirals toward $\mathbf{w}^\ast$ (the distance increases at $48\%$ of the first 600 iterations; it is $0.57$ after $100$ iterations, $7.7 \cdot 10^{-3}$ after $1{,}000$ and $1.9 \cdot 10^{-7}$ after $3{,}000$). With $\beta = 0$ and the heavy-ball step $\eta = 0.733$, which exceeds $2/L = 0.498$, the iteration is plain gradient descent *above its stability limit*: it diverges (the distance exceeds $10^{12}$ at iteration $43$). The step of Theorem 7.8 is stable only *together* with its momentum, which damps the oscillation of the steep directions: momentum is part of the stability of the method, not an optional accelerator.



# %%
def solution_7_9():
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    dg = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(dg.data, dg.target, test_size=0.25, random_state=SEED, stratify=dg.target)
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    n, d = Z.shape
    K = 10
    Y = np.eye(K)[ytr]
    lam = 1 / (3.16 * n)

    def obj(w, idx=None):
        W, b = w[:K * d].reshape(K, d), w[K * d:]
        A, Yb = (Z, Y) if idx is None else (Z[idx], Y[idx])
        S = A @ W.T + b
        logp = S - logsumexp(S, axis=1, keepdims=True)
        P = np.exp(logp)
        f = -np.mean(np.sum(Yb * logp, axis=1)) + 0.5 * lam * np.sum(W**2)
        return f, np.r_[((P - Yb).T @ A / len(A) + lam * W).ravel(), (P - Yb).mean(axis=0)]

    w0 = np.zeros(K * d + K)
    res = minimize(obj, w0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-12, "ftol": 1e-16})
    f_star = res.fun
    L = np.linalg.eigvalsh(Z.T @ Z / n)[-1] / 2 + lam

    def sgd(batch, eta, epochs=10):
        rng = np.random.default_rng(SEED)
        w = w0.copy()
        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                w = w - eta * obj(w, idx=perm[start:start + batch])[1]
        f = obj(w)[0]
        return f - f_star if np.isfinite(f) else np.inf

    batches = [1, 4, 16, 64, 256]
    etaLs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    grid = pd.DataFrame(index=batches, columns=etaLs, dtype=float)
    with np.errstate(all="ignore"):
        for B in batches:
            for e in etaLs:
                grid.loc[B, e] = sgd(B, e / L)
    print("suboptimality after 10 epochs (rows: batch size, columns: eta * L)")
    print(grid.to_string(float_format=lambda v: f"{v:.2e}"))
    best = grid.idxmin(axis=1).astype(float)
    best_gap = grid.min(axis=1)
    print("best eta * L:", dict(zip(batches, [float(v) for v in best])))
    slope = np.polyfit(np.log(batches), np.log(best.values), 1)[0]
    print(f"(b) slope of log(best eta L) against log(B): {slope:.2f}")
    print("(c) best suboptimality after 10 epochs:", "  ".join(f"B={B}: {g:.2e}" for B, g in best_gap.items()),
          f"  | updates per epoch: " + ", ".join(f"{int(np.ceil(n / B))}" for B in batches))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].loglog(batches, best.values, "o-", color=BLUE, label="best $\\eta L$")
    axes[0].loglog(batches, best.values[0] * np.array(batches), "--", color=VERMILION, label="proportional to $B$")
    axes[0].set(xlabel="batch size $B$", ylabel=r"best $\eta L$", title="(a) the best step size")
    axes[0].legend(frameon=False)
    axes[1].loglog(batches, best_gap.values, "o-", color=GREEN)
    axes[1].set(xlabel="batch size $B$", ylabel="suboptimality after 10 epochs", title="(b) the best result")
    fig.tight_layout()
    plt.show()


solution_7_9()

# %% [markdown]
# The best step sizes, in units of $1/L$, are $0.1$, $0.3$, $1$, $10$ and $30$ for $B = 1, 4, 16, 64, 256$, and the slope of their log-log fit is $1.08$: the best step is roughly proportional to the batch size, the linear scaling rule of Section 7.6. The grid has a spacing of a factor $3$ to $3.2$, so each best value is uncertain by that factor, and only the trend is established ($B$ multiplied by $256$, best step by $300$); in particular the jump between $B = 16$ and $B = 64$ ($1 \to 10$) is a grid effect, as the nearly equal entries at $\eta L = 3$ and $10$ for $B = 64$ show ($3.7 \cdot 10^{-2}$ and $3.0 \cdot 10^{-2}$). (c) The best suboptimalities after ten epochs, $1.1 \cdot 10^{-2}$, $1.7 \cdot 10^{-2}$, $2.2 \cdot 10^{-2}$, $3.0 \cdot 10^{-2}$ and $1.1 \cdot 10^{-2}$, all lie within a factor of three of one another, which a single seed and a coarse grid cannot resolve, while the number of updates per epoch varies from $1{,}347$ ($B = 1$) to $6$ ($B = 256$): the larger step compensates for the fewer updates, up to $B = 256$ on this problem. All the runs cost the same number of gradient evaluations (ten epochs), so on a vector machine the large-batch runs are the cheaper ones in time, because a batch gradient is computed with a matrix product; this is why large batches are used in practice, within the limits of the linear scaling rule (Goyal et al., 2017, in the further reading). Beyond $\eta L \approx 30$ the large batches get worse again, as the last columns of the table show.



# %%
def solution_7_10():
    from scipy.special import expit
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    bc = load_breast_cancer(as_frame=True)
    X, y = bc.data.to_numpy(), (1 - bc.target).to_numpy().astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    A, Ate = np.c_[np.ones(len(ytr)), sc.transform(Xtr)], np.c_[np.ones(len(yte)), sc.transform(Xte)]
    s = 2 * ytr - 1
    svm = LinearSVC(C=1e6, loss="hinge", fit_intercept=False, dual=True, max_iter=5_000_000, tol=1e-10).fit(A, ytr)
    w_hat = svm.coef_.ravel()
    m = s * (A @ w_hat)
    acc = lambda w: np.mean(((Ate @ w) > 0) == (yte == 1))
    print(f"(a) separable: {m.min() > 0}; smallest functional margin {m.min():.6f}; ||w_hat|| = {np.linalg.norm(w_hat):.3f}; "
          f"active constraints: {int(np.sum(m < 1 + 1e-6))}; hard-margin test accuracy {acc(w_hat):.4f}")
    n = len(ytr)
    L = np.linalg.eigvalsh(A.T @ A / n)[-1] / 4
    w = np.zeros(A.shape[1])
    rows = []
    for k in range(1, 1_000_001):
        w = w - (A.T @ (expit(A @ w) - ytr) / n) / L
        if k in (10, 100, 1000, 10_000, 100_000, 1_000_000):
            cos = w @ w_hat / (np.linalg.norm(w) * np.linalg.norm(w_hat))
            rows.append({"k": k, "norm": np.linalg.norm(w), "angle (deg)": np.degrees(np.arccos(min(1.0, cos))),
                         "margin ratio": (s * (A @ w)).min() / np.linalg.norm(w) * np.linalg.norm(w_hat),
                         "train loss": np.mean(np.logaddexp(0, A @ w) - ytr * (A @ w)), "test accuracy": acc(w)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


solution_7_10()

# %% [markdown]
# The Breast Cancer training set is separable: the hard-margin classifier has $\lVert\hat{\mathbf{w}}\rVert = 28.4$, $24$ active constraints, and a test accuracy of $0.930$ ($133$ of $143$ patients). Gradient descent on the unpenalized logistic loss approaches this direction extremely slowly: the angle is $71^\circ$ after $10$ steps, $38^\circ$ after $10^4$ and $9.3^\circ$ after $10^6$; the geometric margin is *negative* (the training set is not yet separated) until a number of steps between $10^5$ and $10^6$, and equals $63\%$ of the maximum at $10^6$. The norm grows from $29$ to $76$ in the last decade ($+46$, approaching the $+65$ predicted by $\lVert\hat{\mathbf{w}}\rVert\ln 10$), and the training loss is still $2.6 \cdot 10^{-3}$. This is the $O(1/\ln k)$ of Theorem 7.14, much slower than on Iris (Section 7.7.3), where the margin was at $96\%$ of the maximum after $10^3$ steps and at $99\%$ after $10^4$: with $31$ parameters and $426$ points the data separate with a thin margin and the limit is reached only after astronomical times.
# (c) The test accuracy of the iterates is $0.972$, $0.986$, $0.965$, $0.965$, $0.937$ and $0.930$ at $k = 10, \dots, 10^6$: it falls as the iterate approaches the hard-margin classifier and ends at its value ($0.930$). The early-stopped iterates are *better* than the limit, by $5$ to $8$ patients out of $143$, the same lesson as Section 7.7.2: stopping early regularizes (Proposition 7.13), and the implicit bias of the limit, the maximum-margin classifier fitted without penalty in $31$ dimensions with $24$ active points, is not the best bias for these data (the unpenalized fit also had a test log-loss of $2.17$ against $0.078$ at $C = 1$ in Section 6.4.4). Two cautions: the test set is small, and the differences among the first four accuracies ($0.965$ to $0.986$, up to three patients) are within the noise of the evaluation; the difference to the limit is larger, but it comes from one split, and the number of steps would have to be chosen on validation data (Chapter 5), not on the test set.
