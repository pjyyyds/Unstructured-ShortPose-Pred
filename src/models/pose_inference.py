"""Optional deployment transform for an already loaded, evaluated pose predictor."""
import copy

import torch
from torch.fx import symbolic_trace

from .pose_stgcn import MaskedSTGCNBlock
from .predictor import UnstructuredShortPosePredictor


def _aggregate_before_projection(block: MaskedSTGCNBlock):
    """Move the joint-axis matrix multiply before the bias-free channel expansion."""
    if block.nbr_lin.bias is not None:
        raise ValueError("Joint aggregation commutes only with a bias-free projection here")
    traced = symbolic_trace(block)
    projections = [node for node in traced.graph.nodes if node.op == "call_module" and node.target == "nbr_lin"]
    if len(projections) != 1:
        raise ValueError("Expected one neighbor projection in the pose graph")
    projection = projections[0]
    users = list(projection.users)
    if len(users) != 1 or users[0].target != torch.matmul or users[0].args[1] is not projection:
        raise ValueError("The pose block no longer applies adjacency to the projected joint features")
    previous_product = users[0]
    with traced.graph.inserting_before(projection):
        aggregate = traced.graph.call_function(torch.matmul, (previous_product.args[0], projection.args[0]))
    projection.args = (aggregate,)
    previous_product.replace_all_uses_with(projection)
    traced.graph.erase_node(previous_product)
    traced.graph.lint()
    traced.recompile()
    return traced


def prepare_pose_inference(model: UnstructuredShortPosePredictor) -> UnstructuredShortPosePredictor:
    """Create a deployment copy with identical state keys. Call once, outside timed inference."""
    if model.training:
        raise ValueError("Call eval() on the loaded predictor before preparing inference")
    optimized = copy.deepcopy(model)
    pose = getattr(optimized, "pose_enc", None)
    if pose is None:
        return optimized
    for name, block in list(pose.named_children()):
        if isinstance(block, MaskedSTGCNBlock):
            setattr(pose, name, _aggregate_before_projection(block))
    return optimized.eval()
