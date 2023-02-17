# Owner(s): ["module: inductor"]
import math
import re
import unittest
from types import FunctionType

import torch
import torch._inductor.config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.backends.cuda import sdp_kernel
from torch.fx import GraphModule
import torch._inductor._sdpa_pattern_rewriter as sdpa_pattern_rewriter
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_SDPA, TEST_CUDA
from torch.testing._internal.common_utils import freeze_rng_state, IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.common_cuda import TEST_CUDA, PLATFORM_SUPPORTS_FUSED_SDPA
from torch.backends.cuda import sdp_kernel
import unittest
import math

def _safe_str(target):
    if isinstance(target, FunctionType):
        return target.__name__
    else:
        return re.sub(r"at 0x[a-f0-9]+", "at [address]", str(target))


def create_graph_desc(gm: GraphModule):
    return "\n".join(
        [
            "    ".join(
                [
                    _safe_str(n.op),
                    _safe_str(n.name),
                    _safe_str(n.target),
                    _safe_str(n.args),
                    _safe_str(n.kwargs),
                ]
            )
            for n in gm.graph.nodes
        ]
    )


class TestPatternMatcher(TestCase):
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return torch.add(torch.mm(a, b), torch.mm(c, d))

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

    def test_addmm(self):
        def fn(a, b, c):
            return torch.add(a, torch.mm(b, c)), torch.mm(a, b) + c

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        e1, e2 = fn(*args)
        a1, a2 = torch.compile(fn)(*args)
        torch.testing.assert_close(a1, e1)
        torch.testing.assert_close(a2, e2)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_mm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.mm(a, b),
                    torch.mm(b, c),
                    torch.mm(a, c),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_addmm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                    torch.addmm(c, a, b),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_slice_cat(self):
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            torch.randn(2, 32, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_1(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': 4.0}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                expected = dot_prod_attention(*args)
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)

                    torch.testing.assert_close(actual, expected)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_attn_weights_not_dead(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                attn_weights = (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                )
                return attn_weights.matmul(value), attn_weights

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    transpose    transpose    (arg1, -2, -1)    {}
call_function    matmul    <built-in method matmul of type object at [address]>    (arg0, transpose)    {}
call_method    div    div    (matmul, 4.0)    {}
call_method    softmax    softmax    (div,)    {'dim': -1}
call_method    matmul_1    matmul    (softmax, arg2)    {}
output    output    output    ([matmul_1, softmax],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                expected = dot_prod_attention(*args)
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)

                    torch.testing.assert_close(actual, expected)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_2(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .mul(1.0 / math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 0.25)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                expected = dot_prod_attention(*args)
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)

                    torch.testing.assert_close(actual, expected)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_3(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return torch.nn.functional.dropout(
                    torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                    p=0.4,
                    training=True,
                    inplace=False,
                ).matmul(value)

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.4, 'is_causal': False, 'scale_factor': 3.0}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    with freeze_rng_state():
                        efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        actual = efficient_dot_prod_attention(*args)
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        False
                    )
                    with freeze_rng_state():
                        inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        expected = inefficient_dot_prod_attention(*args)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )
                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_4(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return torch.nn.functional.dropout(
                    torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                    p=0.2,
                    training=True,
                    inplace=False,
                ).matmul(value)

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 0.4)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.2, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                # Now check that the result is identical
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    with freeze_rng_state():
                        efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        actual = efficient_dot_prod_attention(*args)
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        False
                    )
                    with freeze_rng_state():
                        inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        expected = inefficient_dot_prod_attention(*args)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )
                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_5(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(0.1)
                    .masked_fill(attn_mask, float("-inf"))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((1, 1, 8, 8), device="cuda")
                > 0,  # Note the >0 to turn this into a binary mask
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
placeholder    arg3    arg3    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': arg3, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': 0.1}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    with freeze_rng_state():
                        efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        actual = efficient_dot_prod_attention(*args)
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        False
                    )
                    with freeze_rng_state():
                        inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        expected = inefficient_dot_prod_attention(*args)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )
                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_6(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return torch.nn.functional.dropout(
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(0.1)
                    .masked_fill(attn_mask, float("-inf"))
                    .softmax(dim=-1),
                    p=0.3,
                    training=True,
                    inplace=False,
                ).matmul(value)

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((1, 1, 8, 8), device="cuda")
                > 0,  # Note the >0 to turn this into a binary mask
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
placeholder    arg3    arg3    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': arg3, 'dropout_p': 0.3, 'is_causal': False, 'scale_factor': 0.1}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            # These checks are disabled until the feature can be rolled out and graph rewrites can be activated
            if False: # TODO: Reactivate after rollout
                saved_flag = (
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention
                )
                torch._inductor.config.pattern_replace_scaled_dot_product_attention = True
                try:
                    with freeze_rng_state():
                        efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        actual = efficient_dot_prod_attention(*args)
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        False
                    )
                    with freeze_rng_state():
                        inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                        expected = inefficient_dot_prod_attention(*args)
                finally:
                    torch._inductor.config.pattern_replace_scaled_dot_product_attention = (
                        saved_flag
                    )
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
