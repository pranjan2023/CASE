from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, Optional
from core.dag import ExperimentDAG

class SystemState(BaseModel):

    iteration: int = 0

    geometry: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)

    experiment_id: Optional[str] = None

    best_geometry: Optional[Dict[str, Any]] = None

    history: list = Field(default_factory=list)

    best_mean: Optional[float] = None
    best_std: float = 0.0

    last_mean: float = 0.0
    last_std: float = 0.0
    last_n: int = 0

    dag: ExperimentDAG = Field(default_factory=ExperimentDAG)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    p_values: list = Field(default_factory=list)

    multi_stats: Dict[str, Any] = Field(default_factory=dict)
    cross_valid: bool = True