"""Component registry: maps class names <-> classes for dynamic (de)serialisation.

This is the single place that knows the full set of graph encoders and dict
learners. The artifact store reads a component's `class` name from a saved
manifest and asks the registry for the matching class, so adding a new
implementation (e.g. FSM encoder, LC-KSVD learner) means registering it here —
nothing else in the persistence/inference layer changes.

Imports are lazy so that, e.g., loading a WL+FDDL bundle does not drag in an
unrelated learner's heavy dependencies.
"""


def _encoder_loaders():
    def _wl():
        from graph_encoders.wl import WL
        return WL

    def _edge_wl():
        from graph_encoders.wl_edge import EdgeWL
        return EdgeWL

    def _fsm():
        from graph_encoders.fsm import FSM
        return FSM

    def _wl_full():
        # Comparison arm only (pca/): WL with the feature-selection cut disabled.
        from pca.wl_full import WLFull
        return WLFull

    return {
        "WL": _wl,
        "EdgeWL": _edge_wl,
        "FSM": _fsm,
        "WLFull": _wl_full,
    }


def _dict_learner_loaders():
    def _fddl_gpu():
        from dict_learners.fddl_gpu import FDDLGPU
        return FDDLGPU

    def _aksvd():
        from dict_learners.aksvd import AKSVD
        return AKSVD

    def _lcksvd():
        from dict_learners.lcksvd import LCKSVDLearner
        return LCKSVDLearner

    def _frozen_ksvd():
        from dict_learners.frozen_ksvd_learner import FrozenKSVDLearner
        return FrozenKSVDLearner

    def _frozen_ksvd_gpu():
        from dict_learners.frozen_ksvd_learner_gpu import FrozenKSVDLearnerGPU
        return FrozenKSVDLearnerGPU

    def _bayesian():
        from dict_learners.bayesian import BayesianDL
        return BayesianDL

    def _bayesian_gpu():
        from dict_learners.bayesian_gpu import BayesianDLGPU
        return BayesianDLGPU

    def _online_dl():
        from dict_learners.online_dl import OnlineDL
        return OnlineDL

    def _pca_then_dict_learner():
        # Comparison arm only (pca/): a PCA reduction composed with a real
        # dictionary learner, presented to the pipeline as one DictLearner.
        from pca.pca_dict_learner import PCAThenDictLearner
        return PCAThenDictLearner

    return {
        "FDDLGPU": _fddl_gpu,
        "AKSVD": _aksvd,
        "LCKSVDLearner": _lcksvd,
        "FrozenKSVDLearner": _frozen_ksvd,
        "FrozenKSVDLearnerGPU": _frozen_ksvd_gpu,
        "BayesianDL": _bayesian,
        "BayesianDLGPU": _bayesian_gpu,
        "OnlineDL": _online_dl,
        "PCAThenDictLearner": _pca_then_dict_learner,
    }


def get_encoder_class(name: str):
    loaders = _encoder_loaders()
    if name not in loaders:
        raise KeyError(
            f"Unknown graph encoder '{name}'. Registered: {sorted(loaders)}. "
            f"Register it in utils/registry.py."
        )
    return loaders[name]()


def get_dict_learner_class(name: str):
    loaders = _dict_learner_loaders()
    if name not in loaders:
        raise KeyError(
            f"Unknown dict learner '{name}'. Registered: {sorted(loaders)}. "
            f"Register it in utils/registry.py."
        )
    return loaders[name]()
