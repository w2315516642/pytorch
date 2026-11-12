import copy
from collections.abc import Callable
from typing import Any

import torch
from torch._guards import tracing, TracingContext
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_leaves

from .schemas import AOTDispatchCompiler, AOTState


def capture_mutation_phases(
    state: AOTState, flat_fn: Callable[..., Any]
) -> torch.fx.GraphModule | None:
    if (
        not state.needs_autograd
        or state.aot_config.is_export
        or state.aot_config.disable_functionalization
    ):
        return None
    indices = [
        i for i, info in enumerate(state.fw_metadata.input_info)
        if info.requires_grad and not info.is_leaf and info.mutates_data
        and not info.mutations_hidden_from_autograd
        and not info.mutations_under_no_grad_or_inference_mode
    ]
    if not indices:
        return None
    # Raw tracing may change input metadata; keep AOT's tracing inputs intact.
    converter = FakeTensorConverter()
    trace_args = [
        converter.from_real_tensor(state.fake_mode, arg, shape_env=state.fake_mode.shape_env)
        if isinstance(arg, torch.Tensor) else arg
        for arg in state.flat_args
    ]
    graph = make_fx(flat_fn, pre_dispatch=True)(*trace_args)
    placeholders = list(graph.graph.find_nodes(op="placeholder"))
    inputs = [placeholders[i].meta["val"] for i in indices]
    partition, partitions = 0, {}
    grad_enabled = torch.is_grad_enabled()
    for node in graph.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        partitions[node] = partition
        if node.target is torch._C._set_grad_enabled:
            grad_enabled = node.args[0]
            continue
        schema = getattr(node.target, "_schema", None)
        if not grad_enabled or schema is None:
            continue
        writes = [
            node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
            for i, arg in enumerate(schema.arguments)
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        if any(
            isinstance(arg, torch.fx.Node)
            and isinstance(arg.meta.get("val"), torch.Tensor)
            and torch._C._is_alias_of(arg.meta["val"], inp)
            for arg in tree_leaves(writes) for inp in inputs
        ):
            partition += 1
    if len(set(partitions.values())) < 2:
        return None
    split = split_module(
        graph, graph, partitions.__getitem__, keep_original_order=True, tuple_return=True
    )
    sources = state.aot_config.aot_autograd_arg_pos_to_source
    source_by_name = {
        node.name: sources[i] if sources is not None else None
        for i, node in enumerate(placeholders)
    }
    for name in split.meta["partition_names"]:
        for node in split.get_submodule(name).graph.find_nodes(op="placeholder"):
            node._dynamo_source = source_by_name.get(node.name)
    return split


class _CompiledPhase(torch.nn.Module):
    def __init__(self, compiled: Callable[..., Any]) -> None:
        super().__init__()
        self.compiled = compiled

    def forward(self, *args: Any) -> Any:
        return self.compiled(*args)


def compile_mutation_phases(
    state: AOTState,
    split: torch.fx.GraphModule,
    partition_fn: Callable[..., Any],
    fw_compiler: AOTDispatchCompiler,
    bw_compiler: AOTDispatchCompiler | None,
    inference_compiler: AOTDispatchCompiler | None,
) -> Callable[..., Any]:
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

    class CompilePhases(torch.fx.Interpreter):
        def call_module(self, target: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            submod = self.fetch_attr(target)
            num_inputs = len(args)
            num_outputs = len(next(iter(submod.graph.find_nodes(op="output"))).args[0])

            def phase_compiler(compiler: AOTDispatchCompiler) -> AOTDispatchCompiler:
                def compile(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
                    gm.meta["aot_input_mutation_phase"] = (num_inputs, num_outputs)
                    return compiler(gm, example_inputs)
                return compile

            parent = TracingContext.try_get()
            context = copy.copy(parent) if parent is not None else TracingContext(state.fake_mode)
            context.ddp_optimizer_ctx = None
            context.fakify_first_call = True
            with tracing(context):
                compiled = aot_module_simplified(
                    submod, args,
                    fw_compiler=phase_compiler(fw_compiler), bw_compiler=bw_compiler or fw_compiler,
                    partition_fn=partition_fn,
                    decompositions=state.aot_config.decompositions,
                    keep_inference_input_mutations=state.aot_config.keep_inference_input_mutations,
                    inference_compiler=phase_compiler(inference_compiler or fw_compiler),
                    ignore_shape_env=state.aot_config.ignore_shape_env,
                )
                result = compiled(*args, **kwargs)
            self.module.set_submodule(target, _CompiledPhase(compiled))
            return result

    CompilePhases(split).run(*state.flat_args)
    return make_boxed_func(split.forward)
