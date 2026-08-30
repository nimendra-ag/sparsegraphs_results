"""PCA / truncated-SVD reduction of the WL embedding space.

This is the challenger to the elbow/energy feature cut. The distinction that
matters, and the reason this lives in its own class rather than inside the
encoder: the cuts in `utils/elbow.py` *select* -- they return a subset of the
original subtree hashes, so every kept column is still a named WL feature. PCA
*transforms* -- every output column is a dense weighted combination of all
inputs. Nothing is discarded; the vocabulary is rotated, not filtered. A run of
this reducer therefore still has to compute all ~10k subtree hashes for every
graph at inference time, which the energy cut does not.

Two ways to choose the output width:

    n_components=<int>   fixed width. Preferred, because comparing arms at
                         matched dimensionality is the only way to avoid
                         confounding "PCA is better" with "narrower is better".
    energy=<float>       keep the fewest components covering that fraction of
                         total variance. This is structurally the same rule as
                         `find_energy_cut` -- sort a non-negative spectrum,
                         accumulate, stop at the target -- applied to the
                         eigenvalue spectrum instead of the supervised
                         discriminative-score spectrum.

Centering and sparsity: true PCA subtracts the column mean, which turns the WL
count matrix (overwhelmingly zeros) fully dense and makes almost every entry
negative. That is already the case here because the encoder hands us a dense
array, so we pay the memory cost regardless. `center=False` switches to a
truncated SVD instead, which is the variant that would let a future sparse
encoder path stay sparse; report it as SVD, not PCA, if you use it.
"""

import json
import os

import numpy as np
import joblib

# Fitted on the training split only. `singular_values_`/`components_` are the
# learned state; `mean_` too when centering.
_STATE_FILE = "pca_state.joblib"
_CONFIG_FILE = "pca_config.json"


class PCAReducer:
    """Fit a linear projection on the training embeddings; apply it everywhere else.

    Parameters
    ----------
    n_components : int, optional
        Fixed output width. Exactly one of `n_components` / `energy` is required.
    energy : float, optional
        Target fraction of total variance in (0, 1]; the width is the smallest
        number of components reaching it.
    max_components : int, default 2048
        Only used with `energy`: the randomized solver needs a component budget
        up front, so we fit this many and then cut the spectrum. If the target is
        not reached within the budget, `fit` warns and keeps all of them rather
        than silently under-delivering.
    center : bool, default True
        True -> PCA (mean-centered). False -> truncated SVD (no centering).
    whiten : bool, default False
        Rescales components to unit variance. Off by default: it would equalise
        component scales and change the geometry FDDL's reconstruction sees,
        which is a second variable this experiment is not trying to test.
    dtype : str, default "float32"
        The embedding matrix is cast to this before fitting. float32 halves the
        peak memory of the uncut ~10k-wide matrix and matches FDDL, which casts
        its inputs to float32 anyway.
    """

    def __init__(self, n_components=None, energy=None, max_components=2048,
                 center=True, whiten=False, dtype="float32", random_state=42):
        if (n_components is None) == (energy is None):
            raise ValueError(
                "PCAReducer needs exactly one of n_components (fixed width) or "
                "energy (variance target); got "
                f"n_components={n_components!r}, energy={energy!r}."
            )
        if energy is not None and not (0.0 < energy <= 1.0):
            raise ValueError(f"energy must be in (0, 1]; got {energy!r}.")

        self.n_components = n_components
        self.energy = energy
        self.max_components = max_components
        self.center = center
        self.whiten = whiten
        self.dtype = dtype
        self.random_state = random_state

        self._model = None            # fitted sklearn PCA / TruncatedSVD
        self.n_components_ = None     # width actually used by transform()
        self.explained_variance_ratio_ = None   # full fitted spectrum
        self.n_features_in_ = None

    # --- fitting -------------------------------------------------------------

    def _make_model(self, k):
        # Randomized solver throughout: an exact SVD of a ~20k x 10k matrix is
        # not worth the hours, and the randomized one is accurate to well past
        # the precision this comparison can resolve.
        if self.center:
            from sklearn.decomposition import PCA
            return PCA(n_components=k, svd_solver="randomized",
                       whiten=self.whiten, random_state=self.random_state)
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=k, algorithm="randomized",
                            random_state=self.random_state)

    def fit(self, X, y=None):
        """Fit on the training embeddings. `y` is ignored -- PCA is unsupervised.

        That `y` is ignored is the whole experimental point: the baseline spends
        the labels on choosing features, this arm spends nothing and relies on
        variance alone.
        """
        X = np.asarray(X, dtype=self.dtype)
        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        budget = self.n_components if self.n_components is not None else self.max_components
        # Neither solver can return more components than the matrix has rank.
        budget = int(min(budget, n_samples, n_features))
        if budget < 1:
            raise ValueError(f"Cannot fit a reducer on a {X.shape} matrix.")

        kind = "PCA" if self.center else "TruncatedSVD"
        print(f"  [{kind}] fitting on {n_samples} x {n_features} ({self.dtype}), "
              f"budget={budget} components ...", flush=True)

        self._model = self._make_model(budget)
        self._model.fit(X)
        # Exact fractions of the TRUE total variance: sklearn computes the full
        # column variance sum even when the solver is randomized, so these ratios
        # do not sum to 1 and are not renormalised. That is what makes the energy
        # target below mean what it says.
        self.explained_variance_ratio_ = np.asarray(
            self._model.explained_variance_ratio_, dtype=float
        )

        self.n_components_ = (
            budget if self.n_components is not None else self._energy_width(budget)
        )

        covered = float(self.explained_variance_ratio_[:self.n_components_].sum())
        print(f"  [{kind}] using {self.n_components_} components "
              f"covering {100.0 * covered:.2f}% of total variance", flush=True)
        return self

    def _energy_width(self, budget):
        """Smallest prefix of the eigenvalue spectrum reaching `self.energy`.

        Same rule as `utils.elbow.find_energy_cut`, but thresholded against the
        true total variance rather than the sum of the array it is handed --
        `explained_variance_ratio_` is already truncated at `budget`, so summing
        it would move the goalposts.
        """
        cumulative = np.cumsum(self.explained_variance_ratio_)
        if cumulative[-1] < self.energy:
            print(
                f"  [WARN] {budget} components cover only "
                f"{100.0 * cumulative[-1]:.2f}% of variance, short of the "
                f"{100.0 * self.energy:.2f}% target. Keeping all {budget}. "
                f"Raise max_components if you need the target met exactly.",
                flush=True,
            )
            return int(budget)
        return int(np.searchsorted(cumulative, self.energy) + 1)

    # --- applying ------------------------------------------------------------

    def transform(self, X):
        """Project onto the fitted basis, truncated to `n_components_`.

        Slicing the output, rather than re-fitting at the narrower width, avoids
        mutating the fitted sklearn object's internals. The two are equivalent
        for the exact SVD, and equivalent to measurement precision here on the
        front-loaded spectra this pipeline produces. They are not identical in
        general: the randomized solver's accuracy depends on its component
        budget, so on a flat spectrum a fit at `max_components` sliced to k and a
        direct fit at k can disagree in the trailing components (leading ones
        agree to ~1e-3). Where they differ the sliced version is the *better*
        estimate of the true top-k subspace, since it was found with more
        oversampling. This only ever arises with `energy`; with a fixed
        `n_components` the slice is a no-op.
        """
        if self._model is None:
            raise RuntimeError("PCAReducer.transform called before fit().")
        X = np.asarray(X, dtype=self.dtype)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"PCAReducer was fitted on {self.n_features_in_} features but got "
                f"{X.shape[1]}. The encoder vocabulary must be identical between "
                f"fit and transform."
            )
        return self._model.transform(X)[:, :self.n_components_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    # --- persistence ---------------------------------------------------------

    def _config(self):
        return {
            "class": type(self).__name__,
            "n_components": self.n_components,
            "energy": self.energy,
            "max_components": self.max_components,
            "center": self.center,
            "whiten": self.whiten,
            "dtype": self.dtype,
            "random_state": self.random_state,
            # resolved at fit time -- the numbers a reader actually wants
            "n_components_": self.n_components_,
            "n_features_in_": self.n_features_in_,
            "explained_variance_at_cut": (
                float(self.explained_variance_ratio_[:self.n_components_].sum())
                if self.explained_variance_ratio_ is not None else None
            ),
        }

    def save(self, dirpath):
        if self._model is None:
            raise ValueError("PCAReducer has nothing to save; fit it first.")
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(
            {"model": self._model,
             "explained_variance_ratio_": self.explained_variance_ratio_},
            os.path.join(dirpath, _STATE_FILE),
        )
        with open(os.path.join(dirpath, _CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

    @classmethod
    def load(cls, dirpath):
        with open(os.path.join(dirpath, _CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        state = joblib.load(os.path.join(dirpath, _STATE_FILE))

        reducer = cls(
            n_components=config["n_components"],
            energy=config["energy"],
            max_components=config["max_components"],
            center=config["center"],
            whiten=config["whiten"],
            dtype=config["dtype"],
            random_state=config["random_state"],
        )
        reducer._model = state["model"]
        reducer.explained_variance_ratio_ = state["explained_variance_ratio_"]
        # The saved resolved width is authoritative: with an energy target it was
        # decided by the training data, and re-deriving it here could disagree.
        reducer.n_components_ = config["n_components_"]
        reducer.n_features_in_ = config["n_features_in_"]
        return reducer
