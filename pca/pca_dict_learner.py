"""Compose a PCA reduction with a dictionary learner, as a single DictLearner.

`utils/pipeline.py` has exactly two stages -- encoder, then dictionary learner --
with no hook in between, which is where a reduction step naturally belongs. The
cheapest way to insert one without touching the shared pipeline (and so without
risking the baseline's behaviour) is to make the reduction *part of* the thing
the pipeline already calls:

    encoder -> [ PCAReducer -> FDDLGPU ] -> classifiers
               \\_________ this class ________/

Because it satisfies the `DictLearner` contract, `utils/pipeline.py`,
`utils/export.py`, and `utils/artifact_store.py` all work on it unchanged.

Ordering note, and it is a real one: `pipeline.fit_encoder_and_dictionary` drops
all-zero training embeddings *before* calling `fit` here. That is the correct
order and it is worth not disturbing -- the all-zero check is only meaningful in
hash-count space, where a zero row genuinely means "no surviving features".
After centering, no row is zero and the guard would silently stop doing anything.
"""

import json
import os

from dict_learners.dict_learner import DictLearner
from pca.reducer import PCAReducer

_CONFIG_FILE = "pca_pipeline_config.json"
_REDUCER_SUBDIR = "reducer"
_INNER_SUBDIR = "inner"


class PCAThenDictLearner(DictLearner):
    """Reduce the embedding space with PCA, then learn a dictionary in it.

    Parameters
    ----------
    reducer : PCAReducer
        Unfitted. Fitted here on the training embeddings only.
    inner : DictLearner
        The real dictionary learner (e.g. `FDDLGPU`), which sees the reduced
        space and never the original vocabulary.
    """

    def __init__(self, reducer, inner):
        super().__init__(name=f"PCA->{inner.name}")
        self.reducer = reducer
        self.inner = inner

    # --- DictLearner contract -----------------------------------------------

    def fit(self, training_graph_embeddings, y_train=None):
        """Fit the projection on the training embeddings, then the dictionary.

        `y_train` is forwarded untouched: the reducer ignores it (unsupervised),
        the inner learner may require it (FDDL does). So the labels still reach
        the dictionary -- what this arm removes is the labels' influence on
        *feature selection*, not on dictionary learning.
        """
        reduced = self.reducer.fit_transform(training_graph_embeddings)
        print(f"  [PCAThenDictLearner] {training_graph_embeddings.shape} -> "
              f"{reduced.shape}", flush=True)
        self.inner.fit(training_graph_embeddings=reduced, y_train=y_train)
        return self

    def infer(self, infer_graph_embeddings):
        return self.inner.infer(self.reducer.transform(infer_graph_embeddings))

    def n_atoms(self) -> int:
        return self.inner.n_atoms()

    def reduce(self, embeddings):
        """Project encoder-space embeddings into the dictionary's input space.

        Nothing in the current pipeline calls this -- `run_arm_a.py` goes through
        `export_pipeline`, which only ever needs `fit`/`infer`. It exists for
        `SRCClassifier`, which scores against *encoder-space* rather than
        sparse-code inputs: for this arm the dictionary lives in the reduced
        space, so SRC would have to be fed `reduce(embeddings)` explicitly or its
        residuals are computed against the wrong basis (and `D @ z` would not
        even be conformable). That wiring lives in the MC-CV driver, which this
        arm deliberately does not have.
        """
        return self.reducer.transform(embeddings)

    # --- attributes the SRCClassifier duck-type contract expects -------------
    # Forwarded explicitly rather than via __getattr__ so a typo surfaces as an
    # AttributeError instead of silently resolving somewhere unexpected.

    @property
    def classes_(self):
        # Also read by artifact_store.save_bundle when writing the manifest.
        return getattr(self.inner, "classes_", None)

    @property
    def D(self):
        return getattr(self.inner, "D", None)

    @property
    def k(self):
        return getattr(self.inner, "k", None)

    @property
    def M_i(self):
        return getattr(self.inner, "M_i", None)

    @property
    def device(self):
        return getattr(self.inner, "device", None)

    @property
    def lambda1(self):
        return getattr(self.inner, "lambda1", 0.1)

    @property
    def ipm_iters(self):
        return getattr(self.inner, "ipm_iters", 15)

    # --- persistence ---------------------------------------------------------

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        # Each component serialises itself into its own subdirectory, mirroring
        # how the artifact store treats encoder/ and dict_learner/.
        self.reducer.save(os.path.join(dirpath, _REDUCER_SUBDIR))
        self.inner.save(os.path.join(dirpath, _INNER_SUBDIR))
        with open(os.path.join(dirpath, _CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class": type(self).__name__,
                    "name": self.name,
                    "reducer_class": type(self.reducer).__name__,
                    "inner_class": type(self.inner).__name__,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, dirpath: str) -> "PCAThenDictLearner":
        from utils.registry import get_dict_learner_class

        with open(os.path.join(dirpath, _CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)

        reducer = PCAReducer.load(os.path.join(dirpath, _REDUCER_SUBDIR))
        inner_cls = get_dict_learner_class(config["inner_class"])
        inner = inner_cls.load(os.path.join(dirpath, _INNER_SUBDIR))
        return cls(reducer=reducer, inner=inner)
