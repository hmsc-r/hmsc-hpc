# Marginalized Rho Sampler for HMSC

**Session Date:** 2026-01-11
**Branch:** `fastVectorRho`
**Status:** Mathematical derivation complete, implementation pending

---

## 1. Problem Statement

### Current Approach
The current `updateRhoInd` function (`hmsc/updaters/updateRhoInd.py`) samples $\rho$ **conditional on $\beta$**:
$$p(\rho | \beta, \Gamma, V, \ldots)$$

This creates coupling between $\rho$ and $\beta$ that can hurt mixing efficiency.

### Goal
Sample $\rho$ from the **marginalized** distribution:
$$p(\rho | Z, \Gamma, V, X, \Lambda, \eta, \sigma) \propto p(\rho) \int p(Z | \beta) p(\beta | \rho) \, d\beta$$

This integrates out $\beta$, improving mixing while maintaining the `phyloFast` tree-based computation (avoiding $O(n_s^3)$ eigendecomposition).

### Key Constraint
Must work within `phyloFast` framework using tree traversal. The insight is that `updateBetaLambda` already handles $X^\top D^{-1} X$ within the tree structure via `phyloFastSampleBatched`.

---

## 2. Model Setup

### Likelihood
$$Z | \beta \sim \mathcal{N}(X\beta + L, D)$$

where:
- $Z \in \mathbb{R}^{n_y \times n_s}$ — latent data
- $X \in \mathbb{R}^{n_y \times n_c}$ — covariates
- $\beta \in \mathbb{R}^{n_c \times n_s}$ — species niches
- $L$ — latent factor contributions ($\sum_r \Pi_r \eta_r \Lambda_r + L_{\text{off}}$)
- $D$ — diagonal covariance (with `iD` $= D^{-1}$ being precision in code)

### Prior on β
$$\text{vec}(\beta) \sim \mathcal{N}(\text{vec}(\mu), \Sigma_\beta)$$

where:
- $\mu = \Gamma T^\top \in \mathbb{R}^{n_c \times n_s}$ — prior mean
- $\Sigma_\beta = C \otimes V_1 + I \otimes V_2$ — prior covariance with phylogenetic structure
- $V_1 = D_\rho^{1/2} V D_\rho^{1/2}$, $V_2 = D_{1-\rho}^{1/2} V D_{1-\rho}^{1/2}$
- $D_\rho = \text{diag}(\rho_1, \ldots, \rho_{n_c})$
- $C$ — phylogenetic correlation matrix

---

## 3. Mathematical Derivation

### Marginal Distribution
Let $S = Z - L$ (data minus latent factors). Marginalizing $\beta$:
$$S | \rho \sim \mathcal{N}(X\mu, \Sigma_Z)$$

where:
$$\Sigma_Z = X \Sigma_\beta X^\top + D$$

### Log Marginal Likelihood
$$\log p(S | \rho) = -\frac{n_y n_s}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma_Z| - \frac{1}{2}r^\top \Sigma_Z^{-1} r$$

where $r = S - X\mu$.

### Applying Woodbury Identity

**On the inverse:**
$$\Sigma_Z^{-1} = (X\Sigma_\beta X^\top + D)^{-1} = D^{-1} - D^{-1}X\Sigma_{\text{post}}X^\top D^{-1}$$

where:
$$\Sigma_{\text{post}} = (\Sigma_\beta^{-1} + X^\top D^{-1} X)^{-1}$$

**Quadratic form:**
$$r^\top \Sigma_Z^{-1} r = r^\top D^{-1} r - w^\top \Sigma_{\text{post}} w$$

where:
$$w = X^\top D^{-1} r = X^\top D^{-1}(S - X\mu)$$

**Log determinant (Matrix Determinant Lemma):**
$$\log|\Sigma_Z| = \log|D| + \log|\Sigma_\beta| - \log|\Sigma_{\text{post}}|$$

### Final Formula for Comparing $\rho$ Values

$$\boxed{\mathcal{L}(\rho) = -\frac{1}{2}\log|\Sigma_\beta| + \frac{1}{2}\log|\Sigma_{\text{post}}| + \frac{1}{2}w^\top \Sigma_{\text{post}} w}$$

where:
- $w = X^\top D^{-1}S - X^\top D^{-1}X \cdot \Gamma T^\top$
- $\Sigma_{\text{post}}^{-1} = \Sigma_\beta^{-1} + X^\top D^{-1}X$

---

## 4. Computational Structure

### Term 1: Prior Terms
$\log|\Sigma_\beta|$ — computed via existing `pfBilinearDet`

Note: $\log|\Sigma_\beta| = n_s \log|V| + n_c \log|Q|$ where $\log|Q|$ comes from `pfBilinearDet`

### Term 2: Posterior Terms
$\log|\Sigma_{\text{post}}|$ and $w^\top \Sigma_{\text{post}} w$ — require **new tree traversal**

**Key insight:** $X^\top D^{-1} X$ is **species-diagonal** (shape $[n_c, n_c, n_s]$), so it enters at tree leaves. This is exactly how `phyloFastSampleBatched` works.

### Required New Function
`phyloFastMarginalLikBatched` — hybrid of:
- `pfBilinearDet` (determinant tracking)
- `pfSample` (incorporating likelihood at leaves)

---

## 5. Why Tree Traversal Yields Correct Results

### Recursive Structure
The phylogenetic covariance has hierarchical structure matching the tree. For a node with children subtrees $A$, $B$:

$$\Sigma_\beta = \begin{pmatrix} \Sigma_{AA} & \Sigma_{AB} \\ \Sigma_{BA} & \Sigma_{BB} \end{pmatrix}$$

Cross-block covariance $\Sigma_{AB}$ has low rank (through common ancestor).

### Schur Complement for Determinants
$$\log|M| = \log|M_{BB}| + \log|M_{AA} - M_{AB}M_{BB}^{-1}M_{BA}|$$

Tree traversal computes this bottom-up:
1. **Leaves:** Start with likelihood precision $X^\top D_j^{-1} X$
2. **Internal nodes:** Apply Schur complement to aggregate children
3. **Root:** Obtain final log determinant

### Woodbury Updates at Each Level

**Incorporate $V_2$ component (non-phylogenetic):**
$$\Pi_{\text{new}} = \Pi - \Pi D_2 (V^{-1} + D_2 \Pi D_2)^{-1} D_2 \Pi$$

**Incorporate $V_1$ component (phylogenetic, going up tree):**
$$\Pi_{\text{parent}} = \Pi_{\text{child}} - t \cdot \Pi D_1 (V^{-1} + t \cdot D_1 \Pi D_1)^{-1} D_1 \Pi$$

### Determinant Accumulation
```
logDetList[d+1] = logDetSum + logDetV + 2*sum(log(diag(LW)))
```

This correctly accumulates:
- Children contributions (logDetSum)
- $\log|V|$ factor (logDetV)
- Schur complement correction (from Cholesky of $W$)

### Quadratic Form Computation
**Going-up phase:** Transforms $w$ through Schur complement operations
**Going-down phase:** Computes conditional means
**Final:** $w^\top \Sigma_{\text{post}} w = \sum_j w_j^\top (\Pi_j^{\text{final}})^{-1} w_j^{\text{final}}$

---

## 6. Algorithm Summary

For each candidate $\rho_k$ value (fixing $\rho_{-k}$):

1. **Compute $w$:**
   ```python
   w = XTiDS - XTiDX @ (Gamma @ T.T)
   # where XTiDS = X.T @ iD @ S, XTiDX = X.T @ iD @ X
   ```

2. **Compute prior terms** via `pfBilinearDet`:
   ```python
   _, logDetPrior = pfBilinearDet(tree, Mu_arr, Mu_arr, root, iV, rhoVec)
   ```

3. **Compute posterior terms** via new `pfMarginalLik`:
   ```python
   logDetPost, wSigmaPostW = pfMarginalLik(tree, root, V, iV, rho, rho2Mat, XTiDX, w)
   ```

4. **Evaluate log likelihood:**
   ```python
   logLik = -0.5 * logDetPrior + 0.5 * logDetPost + 0.5 * wSigmaPostW
   ```

5. **Sample from categorical:**
   ```python
   rhoInd_new = tf.random.categorical(logLik + log(rhopw[:,1]), 1)
   ```

---

## 7. Complexity Analysis

**Current algorithm (conditional on $\beta$):**
$$O(n_c \times \log(n_s) \times n_s \times g_N \times n_c^3) = O(g_N \times n_s \times n_c^4 \times \log n_s)$$

**Marginalized algorithm:**
- Prior terms: $O(\text{depth} \times n_s \times g_N \times n_c^3)$
- Posterior terms: $O(\text{depth} \times n_s \times g_N \times n_c^3)$ (same structure as `pfSample`)

**Total:** Same asymptotic complexity, but better mixing properties.

---

## 8. Relevant Code Files

| File | Purpose |
|------|---------|
| `hmsc/updaters/updateRhoInd.py` | Current $\rho$ sampler (to be modified) |
| `hmsc/updaters/updateBetaLambda.py` | Shows how `pfSample` incorporates $X^\top D^{-1} X$ |
| `hmsc/utils/phylo_fast_utils.py` | Contains `pfBilinearDet`, `pfSample` (add new `pfMarginalLik`) |
| `hmsc/gibbs_sampler.py` | Main Gibbs sampler loop |

---

## 9. Implementation Tasks

- [ ] Create `phyloFastMarginalLikBatched` function in `phylo_fast_utils.py`
  - Follows `pfSample` structure for going-up/going-down
  - Tracks log determinant like `pfBilinearDet`
  - Returns `logDetPost, wSigmaPostW` instead of sample

- [ ] Create `updateRhoIndMarg` function in new file or modify `updateRhoInd.py`
  - Computes $w = X^\top D^{-1}(S - X\mu)$
  - Calls `pfBilinearDet` for prior terms
  - Calls `pfMarginalLik` for posterior terms
  - Samples from categorical

- [ ] Add tests comparing marginal vs conditional samplers

---

## 10. Key Equations Reference

| Quantity | Formula |
|----------|---------|
| Marginal covariance | $\Sigma_Z = X\Sigma_\beta X^\top + D$ |
| Posterior precision | $\Sigma_{\text{post}}^{-1} = \Sigma_\beta^{-1} + X^\top D^{-1} X$ |
| Transformed residual | $w = X^\top D^{-1}(S - X\mu)$ |
| Log marginal lik (for $\rho$) | $\mathcal{L}(\rho) = -\frac{1}{2}\log|\Sigma_\beta| + \frac{1}{2}\log|\Sigma_{\text{post}}| + \frac{1}{2}w^\top \Sigma_{\text{post}} w$ |
| Woodbury inverse | $(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$ |
| Matrix det lemma | $|A + UCV| = |C| \cdot |C^{-1} + VA^{-1}U| \cdot |A|$ |

---

## 11. Session Notes

- User confirmed goal is **mixing efficiency** improvement
- Focus on `phyloFast` pathway (tree-based, no eigendecomposition)
- Vector $\rho$ case: elements sampled one at a time (Gibbs within Gibbs)
- Non-phyloFast case (eigendecomposition) left as-is for now
- Notation: `iD` in code is precision ($D^{-1}$), $D$ is covariance
