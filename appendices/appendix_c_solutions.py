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
