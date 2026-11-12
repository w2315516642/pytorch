# Owner(s): ["module: dynamo"]
import torch
from functorch.compile import nop
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_function
from torch.utils._pytree import tree_leaves, tree_map_only
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def compile_fn(fn, backend, *, dynamic=False):
    if backend == "aot_function":
        return aot_function(fn, fw_compiler=nop, dynamic=dynamic)
    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=dynamic)


class TestAOTMutationPhases(TestCase):
    @parametrize("backend", ["aot_eager", "inductor", "aot_function"])
    @parametrize(
        "kind",
        [
            "basic",
            "view",
            "old_use",
            "aliased_inputs",
            "dependent_mutations",
            "mixed_grad",
            "returned_alias",
            "duplicate",
        ],
    )
    def test_retained_grad(self, device, backend, kind):
        def fn(x, z):
            old = x * 7 if kind == "old_use" else 0
            if kind == "mixed_grad":
                with torch.no_grad():
                    z.mul_(2)
            target = x.view(2, -1) if kind == "view" else x
            target.mul_(3)
            if kind == "dependent_mutations":
                z.add_(x)
            y = x.square() + z.sin() + old
            return (y, x.view(2, -1)) if kind == "returned_alias" else y

        def run(call, count, when):
            leaf = torch.arange(
                1, count + 1, device=device, dtype=torch.float64, requires_grad=True
            )
            x = leaf * 2
            z = x if kind == "duplicate" else leaf * 3
            if kind == "aliased_inputs":
                z = x.view_as(x)
            if when == "before":
                x.retain_grad()
                z.retain_grad()
            y = tree_leaves(call(x, z))
            if when == "after":
                x.retain_grad()
                z.retain_grad()
            loss = sum(value.sum() for value in y) + (x.sin().sum() if when == "after" else 0)
            loss.backward(retain_graph=True)
            first = (
                tree_map_only(torch.Tensor, torch.clone, (x.grad, z.grad))
                if when != "none" else None
            )
            loss.backward()
            grads = (x.grad, z.grad) if when != "none" else None
            return y, first, grads, leaf.grad

        compiled = compile_fn(fn, backend, dynamic=True)
        # Reuse one callable across shapes and changes to retention.
        for count, when in [(8, "none"), (12, "before"), (8, "after")]:
            self.assertEqual(run(compiled, count, when), run(fn, count, when))

    @parametrize("backend", ["aot_eager", "inductor", "aot_function"])
    @parametrize(
        "mode",
        [
            "later_mutation",
            "no_grad_mutation",
            "hook",
            "backward_inputs",
            "autograd_grad",
        ],
    )
    def test_autograd_semantics(self, device, backend, mode):
        def fn(x):
            x.mul_(3)
            return x * 4

        def run(call):
            leaf = torch.ones(4, device=device, requires_grad=True)
            x = leaf * 2
            x.retain_grad()
            y = call(x)
            seen = []
            if mode == "later_mutation":
                x.add_(1)
            elif mode == "no_grad_mutation":
                with torch.no_grad():
                    x.add_(1)
            elif mode == "hook":
                x.register_hook(lambda grad: seen.append(grad.clone()) or grad * 2)
            if mode == "autograd_grad":
                return torch.autograd.grad(y.sum(), x)
            y.sum().backward(inputs=(x,) if mode == "backward_inputs" else None)
            return x.grad, leaf.grad, seen

        self.assertEqual(run(compile_fn(fn, backend)), run(fn))

    @parametrize("backend", ["aot_eager", "inductor"])
    @parametrize("reverse", [False, True])
    def test_separate_backwards(self, device, backend, reverse):
        def fn(x, z):
            x.mul_(3)
            z.mul_(4)
            return x.square(), z.sin()

        def run(call):
            a = torch.ones(8, device=device, requires_grad=True)
            b = torch.ones(6, device=device, requires_grad=True)
            x, z = a * 2, b * 2
            x.retain_grad()
            z.retain_grad()
            outputs = call(x, z)
            # Isolate accumulation from AOT's existing None-versus-zero behavior.
            x.grad, z.grad = torch.zeros_like(x), torch.zeros_like(z)
            snapshots = []
            for index in ([1, 0] if reverse else [0, 1]):
                outputs[index].sum().backward(retain_graph=True)
                snapshots.append((x.grad.clone(), z.grad.clone()))
            return snapshots, a.grad, b.grad

        self.assertEqual(run(compile_fn(fn, backend)), run(fn))

    @parametrize("view", [False, True])
    def test_metadata_mutation(self, device, view):
        def fn(x):
            x.transpose_(0, 1)
            x.mul_(3)
            return x.square()

        def run(call):
            leaf = (
                torch.arange(1, 9, device=device, dtype=torch.float64)
                .reshape(2, 4).requires_grad_()
            )
            x = leaf * 2
            if view:
                x = x.view_as(x)
            y = call(x)
            x.retain_grad()
            y.sum().backward()
            return x, y, x.grad, leaf.grad

        self.assertEqual(run(aot_function(fn, fw_compiler=nop)), run(fn))

    @parametrize("mutate_input", [False, True])
    def test_compiled_regions(self, device, mutate_input):
        def fn(x):
            if mutate_input:
                x.mul_(3)
                return x.square()
            other = torch.zeros_like(x)
            other.copy_(x)
            return other.square()

        backend = AotEagerAndRecordGraphs()
        x = torch.ones(8, device=device, requires_grad=True) * 2
        x.retain_grad()
        y = torch.compile(fn, backend=backend, fullgraph=True)(x)
        expected = 2 if mutate_input else 1
        self.assertEqual(len(backend.fw_graphs), expected)
        y.sum().backward()
        self.assertEqual(len(backend.bw_graphs), expected)
        self.assertEqual(x.grad, 2 * x)


instantiate_device_type_tests(TestAOTMutationPhases, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
