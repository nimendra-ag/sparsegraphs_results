"""WL features, FSM/graph2vec-style frequency selection.

The ablation this file exists for: WL and FSM differ in *two* ways at once —
what a feature IS (md5 subtree hash vs ego-network shape signature) and how the
vocabulary is TRIMMED (supervised discriminative score + energy/elbow cut vs a
fixed raw-frequency budget). Comparing `wl_fddl_gpu` against `fsm_fddl_gpu`
therefore cannot attribute a difference to either cause.

This encoder holds the features fixed and swaps only the trim: identical WL
hashing to `graph_encoders.wl.WL` (inherited byte-for-byte, see below), then
FSM's selection — drop anything seen fewer than `min_count` times, rank the rest
by frequency, keep the top `n_vocab`. Run it against `wl_fddl_gpu` and the only
thing that differs is the selection mechanism.

Why a thin subclass rather than a fork (the `pca/wl_full.py` argument): every
step except `create_vocab` is inherited, so the two arms cannot drift apart in
how graphs are checked, how subtrees are hashed, or how the count matrix is
scaled. A fork would let an unrelated edit to `wl.py` silently desynchronise
them.

Differences from `WL` that callers should know about:

  * SELECTION is unsupervised. `labels` is still accepted positionally (
    utils/pipeline.py calls every encoder as
    `generate_training_embeddings(graphs, y)`) but never read, exactly as in
    `graph_encoders.fsm.FSM`. This arm never sees the vocab_train labels before
    the classifier stage.
  * COUNTS ARE RAW OCCURRENCES, not document frequency. `WL.create_vocab`
    counts each subtree hash once per graph it appears in (`majority_df[word]
    += 1` over `Counter` KEYS, wl.py:73-81). This one counts every occurrence,
    so a hash appearing 6x inside one graph contributes 6, not 1 — matching
    `FSM.create_vocab` (fsm.py:123-126) and gensim's `min_count` inside
    Graph2Vec (graph2vec.py:211). That is the definition being ported, so it is
    deliberate; it also means a single large graph can push a hash to the top of
    the vocabulary on its own.
  * ⚠ `n_vocab` MEANS SOMETHING DIFFERENT HERE THAN IN wl.py, AND DEFAULTS OFF.
    In `WL` it is a *fallback*, reached only when the adaptive cut keeps fewer
    than 50 features (wl.py:184-185) — it can never cap a large vocabulary.
    Here it is FSM's hard *cap*: the frequency-ranked list is sliced to
    `n_vocab` (fsm.py:129). `None` (the default) disables it.

    It defaults off because a cap SILENTLY DISABLES the thing this arm exists to
    study. WL emits one hash per node per iteration, so nci_full at
    `wl_iterations=2` yields ~10k distinct hashes and even `min_count=10` leaves
    a few thousand. With `n_vocab=1000` the cap therefore binds at every useful
    threshold, and since `min_count` only ever deletes from the frequency TAIL —
    which the cap has already deleted — the selected vocabulary comes out
    *byte-identical* for every `min_count` up to the count of the 1000th-ranked
    feature (18 on an 8k-graph sample of nci_full). A whole min_count sweep then
    silently trains the same model N times.

    Set it to an int only if you deliberately want a fixed-width "top-N by
    frequency" arm — in which case `min_count` is no longer the variable and
    should not be swept. Whenever it binds, `create_vocab` says so.
  * There is no `selection_scores_` — nothing is scored on a discriminative
    criterion — so the elbow analytics plot in utils/export.py is skipped for
    this arm (it is guarded by `getattr`, export.py:131) and
    `n_scored_features` in an MC-CV manifest is None (mccv.py:387). Both
    already degrade cleanly; FSM established that path.
  * `self.selection` is set to the string "min_count" rather than left as one
    of WL's cut names, so the MC-CV manifest field (mccv.py:395) identifies
    which arm produced a row instead of claiming an energy/elbow cut that never
    ran. `energy` and `min_features` are None for the same reason.

Deliberately UNCHANGED from `WL`, so the comparison stays clean:

  * `_check_graph` still runs (via the inherited `create_wl_hash`), so every
    node still gets the Karate-Club self-loop and the hashes are the same ones
    `wl_fddl_gpu` computes. FSM skips this; we must not, or the features would
    change too.
  * The embedding is still L2-NORMALISED per row, as in wl.py:206-208. FSM
    leaves its matrix as raw counts. Copying that here would change the
    selection *and* the scaling, and no difference in the results could be
    attributed to either.

Subtree identifiers are md5 hex digests (wlkernalsubtree.py), not the builtin
hash(), so like WL this arm carries no PYTHONHASHSEED caveat.
"""

import os
import json
from collections import Counter

import numpy as np

from graph_encoders.wl import WL


class WLMinCount(WL):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = None,
            min_count: int = 2,
            seed: int = 42
    ):
        # WL's constructor sets up the hashing knobs this class inherits. Its
        # three selection knobs are then overwritten below: they parameterise
        # `WL.create_vocab`, which this class does not use.
        super().__init__(
            wl_iterations=wl_iterations,
            attributed=attributed,
            erase_base_features=erase_base_features,
            n_vocab=n_vocab,
            seed=seed,
        )

        self.name = "MinCountWL"
        self.min_count = min_count

        # Recorded in the MC-CV manifest (mccv.py:395-397). "min_count" is the
        # honest value: no score curve was computed, so naming one of WL's cuts
        # here would mislabel every row this arm writes.
        self.selection = "min_count"
        self.energy = None
        self.min_features = None

    def create_vocab(self, corpus, labels=None):
        """FSM's fixed-frequency budget over WL's subtree hashes.

        Replaces `WL.create_vocab` outright — no discriminative score, no
        max-normalisation, no energy/elbow/percentile cut. `labels` is accepted
        for interface parity with the base class (utils/pipeline.py and
        utils/mccv.py call it positionally through
        `generate_training_embeddings`) and is not read.

        As in `WL`, the returned order IS the embedding column order and must be
        preserved on disk.
        """
        global_counts = Counter()
        for doc in corpus:
            # `doc.words` is the flat per-node-per-iteration list from
            # WeisfeilerLehmanHashing.get_graph_features(), WITH repeats.
            # `Counter.update` on a list adds every occurrence, so this is a raw
            # occurrence count, not the per-graph document frequency wl.py
            # builds. That is the whole point of this arm.
            global_counts.update(doc.words)

        frequent_subtrees = {
            word: count for word, count in global_counts.items()
            if count >= self.min_count
        }

        # Ties at the `n_vocab` boundary fall back to first-appearance order in
        # `global_counts`, which is fixed by the corpus order the pipeline hands
        # us and by md5 (process-stable) — so the cut is reproducible across
        # runs. Same tie-break as fsm.py:128.
        sorted_vocab = sorted(
            frequent_subtrees.items(), key=lambda item: item[1], reverse=True
        )
        # `self.n_vocab is None` -> `[:None]` -> no cap, min_count alone decides.
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        print(f"Total Features {len(global_counts)}")
        print(f"selected {len(trimmed_vocab)} from the fixed frequency budget "
              f"(min_count={self.min_count}, "
              f"n_vocab={self.n_vocab if self.n_vocab is not None else 'off'})")

        # Diagnostics for the two ways this trim can quietly stop being the
        # variable under study. Both are cheap and both have already cost a run.
        if self.n_vocab is not None and len(frequent_subtrees) > self.n_vocab:
            # The count threshold was NOT the binding constraint, the cap was.
            # Every min_count at or below `lowest` yields the SAME vocabulary.
            lowest = trimmed_vocab[-1][1] if trimmed_vocab else 0
            print(f"  WARNING: the n_vocab cap is binding "
                  f"({len(frequent_subtrees)} features passed min_count={self.min_count}). "
                  f"min_count has NO effect on the output below {lowest + 1}; "
                  f"set n_vocab=None to let it drive the width.")
        elif trimmed_vocab:
            print(f"  min_count is the binding constraint "
                  f"(lowest kept count {trimmed_vocab[-1][1]})")

        # NOTE: no fallback branch. `WL` widens to `scored_vocab[:n_vocab]` when
        # its adaptive cut keeps < 50 features (wl.py:184-185); there is no
        # analogue here, because this trim is a fixed budget and cannot collapse
        # unexpectedly — if it returns few features, that IS the dataset's
        # answer at this min_count and should be visible, not patched over.
        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus):
        """Numerically identical to `WL.calc_coefficients`, via an index map.

        `WL` scans the whole vocabulary for every document (wl.py:200-204),
        which is O(n_docs * n_vocab) dict lookups. This trim can legitimately
        keep an order of magnitude more features than an energy cut does, and at
        that width the scan dominates the run. Inverting the loop — map each
        document's own words into column indices — touches only the non-zero
        cells and produces the same matrix, exactly as fsm.py:138-149 and
        wl_edge.py already do.

        The L2 row normalisation below is copied verbatim from wl.py:206-208,
        including the zero-row guard, so this arm's scaling matches the WL
        baseline it is being compared against.
        """
        vocab_index = {word: idx for idx, (word, _) in enumerate(self.vocab)}

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        for i, document in enumerate(corpus):
            words_count = Counter(document.words)
            for word, count in words_count.items():
                # Words outside the vocabulary are dropped, which is what the
                # base class's vocabulary-driven scan does implicitly.
                if word in vocab_index:
                    sparse_vector[i, vocab_index[word]] = count

        norms = np.linalg.norm(sparse_vector, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sparse_vector = sparse_vector / norms

        return sparse_vector

    # --- Persistence ---------------------------------------------------------
    # Same contract as WL/FSM: the only *learned* state is `vocab` (the ordered,
    # frequency-trimmed subtree hashes) and its order IS the embedding column
    # order. Distinct filenames from wl.py's, so a WL bundle and a WLMinCount
    # bundle can never be loaded into the wrong class by accident. The stored
    # per-word value is an int occurrence count here, where WL stores a float
    # score — another reason not to share the file name.
    _CONFIG_FILE = "wl_min_count_config.json"
    _VOCAB_FILE = "wl_min_count_vocab.json"

    def _config(self):
        return {
            "class": type(self).__name__,
            "name": self.name,
            "wl_iterations": self.wl_iterations,
            "attributed": self.attributed,
            "erase_base_features": self.erase_base_features,
            "n_vocab": self.n_vocab,
            "min_count": self.min_count,
            "seed": self.seed,
        }

    def save(self, dirpath: str) -> None:
        if self.vocab is None:
            raise ValueError(
                "WLMinCount has no vocab to save; fit the encoder first."
            )
        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath, self._CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(self._config(), f, indent=2)

        # Preserve order; words are md5 hex strings and the stored value is a
        # global occurrence count -> JSON safe.
        vocab_serialisable = [[str(word), int(count)] for word, count in self.vocab]
        with open(os.path.join(dirpath, self._VOCAB_FILE), "w", encoding="utf-8") as f:
            json.dump(vocab_serialisable, f)

    @classmethod
    def load(cls, dirpath: str) -> "WLMinCount":
        with open(os.path.join(dirpath, cls._CONFIG_FILE), encoding="utf-8") as f:
            config = json.load(f)
        with open(os.path.join(dirpath, cls._VOCAB_FILE), encoding="utf-8") as f:
            vocab = [(word, count) for word, count in json.load(f)]

        encoder = cls(
            wl_iterations=config["wl_iterations"],
            attributed=config["attributed"],
            erase_base_features=config["erase_base_features"],
            n_vocab=config["n_vocab"],
            min_count=config["min_count"],
            seed=config["seed"],
        )
        encoder.vocab = vocab
        # The saved vocab length is authoritative for the embedding width.
        encoder.n_vocab = len(vocab)
        return encoder
