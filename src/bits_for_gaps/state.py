"""In-memory run state for the sequential-design loop.

Replaces the paper code's disk-as-state convention (``np.savetxt``/``pickle`` under
``results/{exp_name}/``, read back on the next iteration -- see ``driver_new.py``'s
``adaptiveEntropy.run_model``) with plain in-memory records. Disk output is still
available, but as an *opt-in* checkpoint (``sampler.adaptiveEntropy.run``'s
``checkpoint_dir`` argument), not the mechanism by which state is threaded across
iterations.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class IterationRecord:
    """Everything produced by one sequential-design iteration.

    ``XData``/``yData`` are the *cumulative* design (all points evaluated up to and
    including this iteration's newly selected point), in the physical input/output
    space -- mirrors the paper code's ``activity_data_{iters}`` file contents.
    """

    iteration: int
    XData: np.ndarray
    yData: np.ndarray
    GPmodel: Any  # fitted gpflow.models.GPR
    trace: np.ndarray  # HMC posterior samples, chain 0, constrained
    chains_states: np.ndarray  # all chains, unconstrained
    rhat: np.ndarray
    ess: np.ndarray
    entropy_field: Optional[np.ndarray] = None  # 2-D only (Phase 5 generalizes)
    xStar: Optional[np.ndarray] = None
    max_entropy: Optional[float] = None
    lml_result: Optional[Any] = None  # scipy OptimizeResult, if maximize_lml ran


@dataclass
class RunHistory:
    """The in-memory record of a sequential-design run: one entry per iteration."""

    records: List[IterationRecord] = field(default_factory=list)

    def append(self, record):
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def __iter__(self):
        return iter(self.records)

    @property
    def last(self):
        return self.records[-1]
