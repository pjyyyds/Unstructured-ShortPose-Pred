"""Regression coverage for the optional post-training pose graph transform."""
import unittest

import torch
from torch import nn

from src.models.pose_inference import _aggregate_before_projection, prepare_pose_inference
from src.models.pose_stgcn import MaskedSTGCNBlock, build_coco17_adjacency
from src.models.predictor import UnstructuredShortPosePredictor


class PoseInferenceTest(unittest.TestCase):
    def test_original_keeps_its_graph_and_parameters(self):
        original = UnstructuredShortPosePredictor("M1").eval()
        deployed = prepare_pose_inference(original)
        self.assertIsInstance(original.pose_enc.block1, MaskedSTGCNBlock)
        self.assertIsInstance(deployed.pose_enc.block1, torch.fx.GraphModule)
        self.assertEqual(original.state_dict().keys(), deployed.state_dict().keys())
        for name, parameter in original.named_parameters():
            other = deployed.get_parameter(name)
            self.assertTrue(torch.equal(parameter, other), name)
            self.assertNotEqual(parameter.data_ptr(), other.data_ptr(), name)

    def test_masked_dense_block_preserves_double_precision_output(self):
        torch.manual_seed(23)
        block = MaskedSTGCNBlock(6, 32).double().eval()
        transformed = _aggregate_before_projection(block)
        joints = torch.randn(3, 5, 17, 6, dtype=torch.float64)
        mask = torch.rand(3, 5, 17, 1).ge(0.3).double()
        mask[0].zero_()
        mask[1].fill_(1)
        adjacency = build_coco17_adjacency().double()
        with torch.inference_mode():
            actual = transformed(joints, mask, adjacency)
            expected = block(joints, mask, adjacency)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)

    def test_training_and_bias_guards_and_motion_only_model(self):
        motion = UnstructuredShortPosePredictor("M0")
        with self.assertRaisesRegex(ValueError, "eval"):
            prepare_pose_inference(motion)
        self.assertFalse(prepare_pose_inference(motion.eval()).training)
        block = MaskedSTGCNBlock(6, 32)
        block.nbr_lin = nn.Linear(6, 32, bias=True)
        with self.assertRaisesRegex(ValueError, "bias"):
            _aggregate_before_projection(block)
