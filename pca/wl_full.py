"""WL encoder with the feature-selection cut disabled.

Arm A of the PCA comparison asks: *does unsupervised linear compression of the
FULL subtree vocabulary beat supervised discriminative selection of a subset?*
Answering it honestly requires a WL encoder that scores features exactly as the
production encoder does but then keeps every one of them, so the only thing that
differs between the arms is what happens after scoring.

Why this is a thin subclass rather than a fork of the encoder: the inherited
`create_vocab` is byte-for-byte the one the baseline runs, so the two arms
cannot drift apart in how features are hashed, counted, or scored. All this
class does is select `WL`'s `selection="none"` path and verify the result.

Why `selection="none"` and not `energy=1.0`: an energy target of 1.0 looks like
it should keep everything, but features scoring exactly 0 (equally present in
both classes) make the cumulative-score curve plateau before the last rank, so
`find_energy_cut` stops early and silently drops them. MUTAG has such features
(139 of 143 kept on seed 41); nci_full does not, which is exactly the kind of
dataset-dependent trap that would have gone unnoticed on the main dataset.
"""

from graph_encoders.wl import WL


class WLFull(WL):
    """WL that keeps its entire scored vocabulary (no elbow/energy/percentile cut).

    Scores every feature identically to `WL` -- the ranking, the imbalance-aware
    `total_presence * |sqrt(p_maj) - sqrt(p_min)|` score, and the max-normalisation
    are all inherited untouched -- then keeps all of them. `selection_scores_` is
    still populated, so the elbow analytics suite renders for this arm too and
    the baseline's cut points stay visible on the figure for reference.
    """

    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50,
            selection: str = "none",
            energy: float = 1.0,
            seed: int = 42,
    ):
        # `selection`/`energy` are accepted only so the inherited `WL.load`
        # classmethod (which forwards every saved config key) can rebuild this
        # class. They are deliberately overridden: this encoder never cuts.
        super().__init__(
            wl_iterations=wl_iterations,
            attributed=attributed,
            erase_base_features=erase_base_features,
            n_vocab=n_vocab,
            min_features=min_features,
            selection="none",
            energy=1.0,
            seed=seed,
        )

    def create_vocab(self, corpus, labels):
        """Score exactly as `WL` does, then verify nothing was dropped.

        The check is the point: "no cut" is a claim the experiment depends on, so
        it fails loudly here rather than silently producing a narrower vocabulary
        that would make Arm A a comparison against the wrong thing.
        """
        vocab = super().create_vocab(corpus, labels)

        n_scored = len(self.selection_scores_)
        if len(vocab) != n_scored:
            raise RuntimeError(
                f"WLFull kept {len(vocab)} of {n_scored} features -- the cut was "
                f"NOT disabled, so this run would compare PCA against a partially "
                f"filtered vocabulary rather than the full one. Check that "
                f"selection='none' survived construction (it is {self.selection!r})."
            )

        print(f"WLFull: keeping the full vocabulary ({len(vocab)} features, no cut)")
        return vocab
