from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, Optional, List
from core.dag import ExperimentDAG


class EpistemicFlags(BaseModel):
    """Emitted once per experiment run. Tracks what we know vs. don't."""
    kernel_subordination: bool = True
    grammar_bounded_search: bool = True
    search_space_contains_optimum: str = "UNKNOWN"
    cross_physics_consistent: bool = True
    synthetic_noise_injected: bool = False     # Phase 2 legacy was True; Phase 3: False
    rejection_reason: Optional[str] = None
    anderson_darling_stat: Optional[float] = None
    jackknife_std: Optional[float] = None


class RejectedExperiment(BaseModel):
    """Full record of a rejected config — fed back into the planner prompt."""
    geometry: Dict[str, Any]
    mean: float
    std: float
    rejection_reason: str   # "no_improvement" | "cross_physics_divergence" | "bh_gate"
    cross_physics_stat: Optional[float] = None
    best_mean_at_rejection: Optional[float] = None


class SystemState(BaseModel):

    iteration: int = 0
    mode: str = "phase3"   # "phase2" (geant4) | "phase3" (hzz oracle)

    geometry: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)

    experiment_id: Optional[str] = None

    best_geometry: Optional[Dict[str, Any]] = None

    history: List[Dict[str, Any]] = Field(default_factory=list)
    rejected: List[RejectedExperiment] = Field(default_factory=list)

    best_mean: Optional[float] = None
    best_std: float = 0.0
    best_jackknife_std: float = 0.0   # for p-value calculations (jackknife std of best)

    last_mean: float = 0.0
    last_std: float = 0.0
    last_jackknife_std: float = 0.0   # for p-value calculations (jackknife std of current)
    last_n: int = 0

    dag: ExperimentDAG = Field(default_factory=ExperimentDAG)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    p_values: list = Field(default_factory=list)

    multi_stats: Dict[str, Any] = Field(default_factory=dict)
    cross_valid: bool = True

    epistemic_flags: List[EpistemicFlags] = Field(default_factory=list)

    # Phase 3 specific settings (with defaults)
    oracle_runs: int = 20
    oracle_events: int = 1000
    pileup: bool = False
    use_mock_oracle: bool = False 