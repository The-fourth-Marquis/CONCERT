"""NicheFlow-v1.5 reference-only nonlinear transport prototype."""

from .local_residual_flow import LocalResidualFlowResult, fit_predict_local_residual_flow
from .local_residual_flow_predict import predict_task_local_residual_flow
from .factorized_flow import FactorizedFlowResult, fit_predict_factorized_flow
from .factorized_flow_predict import predict_task_factorized_flow
from .anchor_guided_factorized import AnchorGuidedFactorizedResult, fit_predict_anchor_guided_factorized
from .anchor_guided_factorized_predict import predict_task_anchor_guided_factorized
from .anchor_flow import AnchorFlowResult, fit_predict_anchor_residual_flow
from .anchor_flow_predict import predict_task_anchor_residual_flow
from .flow import ConditionalFlowResult, fit_predict_conditional_flow
from .flow_predict import predict_task_conditional_flow
from .graph import build_niche_adjacency
from .model import NonlinearTransportResult, fit_predict_nonlinear_transport
from .predict import predict_task_reference_only
from .transport import entropic_sinkhorn

__all__ = [
    "AnchorFlowResult",
    "LocalResidualFlowResult",
    "FactorizedFlowResult",
    "AnchorGuidedFactorizedResult",
    "ConditionalFlowResult",
    "NonlinearTransportResult",
    "build_niche_adjacency",
    "entropic_sinkhorn",
    "fit_predict_anchor_residual_flow",
    "fit_predict_local_residual_flow",
    "fit_predict_factorized_flow",
    "fit_predict_anchor_guided_factorized",
    "fit_predict_conditional_flow",
    "fit_predict_nonlinear_transport",
    "predict_task_anchor_residual_flow",
    "predict_task_local_residual_flow",
    "predict_task_factorized_flow",
    "predict_task_anchor_guided_factorized",
    "predict_task_conditional_flow",
    "predict_task_reference_only",
]
