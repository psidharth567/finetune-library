"""Multi-GPU check: expert-parallel MoE forward/backward vs a local reference.

Stacks checkpointed MoE layers (learned top-k routing, residuals) and compares
the output and input gradient of the expert-parallel path against the
non-EP grouped computation with the same accumulation order. Expected with the
native all-to-all backend: out_rel == 0 (bitwise), grad_rel <~ 1e-2. DeepEP
combines in bf16, so expect out_rel ~2e-2 there.

    torchrun --nproc-per-node 8 tests/multigpu/ep_gradcheck.py EP_SIZE {native|deepep}
"""
import os, sys
import torch, torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.distributed.device_mesh import init_device_mesh
from finetune_library.moe_parallel import build_moe_layout, expert_parallel_forward

EP, BACKEND = int(sys.argv[1]), sys.argv[2]
CKPT = os.environ.get("GC_CKPT", "1") == "1"
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dist.init_process_group("nccl"); dev = torch.device("cuda")
E, K, H, F, T, L = 256, 8, 2048, 64, 2048, 4
torch.manual_seed(0)
W1 = [(torch.randn(E, F, H, device=dev) * 0.05).to(torch.bfloat16) for _ in range(L)]
W2 = [(torch.randn(E, H, F, device=dev) * 0.05).to(torch.bfloat16) for _ in range(L)]
R = [(torch.randn(H, E, device=dev) * 0.05).to(torch.bfloat16) for _ in range(L)]
torch.manual_seed(100 + rank)
x0 = torch.randn(T, H, device=dev).to(torch.bfloat16)
probe = torch.randn(T, H, device=dev).to(torch.bfloat16)

def grouped(h, ids, wts, w1, w2, offset):
    if h.numel() == 0:
        return h
    local = ids - offset
    perm = torch.argsort(local); inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=h.device)
    offs = torch.bincount(local[perm], minlength=w1.shape[0]).cumsum(0, dtype=torch.int32)
    a = torch._grouped_mm(h[perm], w1.transpose(-2, -1), offs=offs)
    y = torch._grouped_mm(torch.nn.functional.silu(a), w2.transpose(-2, -1), offs=offs)
    return (y * wts[perm].unsqueeze(-1).to(y.dtype))[inv]

def route(h, r):
    p = (h @ r).float().softmax(-1); w, i = p.topk(K, -1); w = w / w.sum(-1, keepdim=True)
    return w.to(torch.bfloat16), i

def ref_layer(h, l):
    w, i = route(h, R[l]); tok = torch.arange(T, device=dev).repeat_interleave(K)
    per = grouped(h[tok], i.reshape(-1), w.reshape(-1), W1[l], W2[l], 0).view(T, K, H)
    per = per.gather(1, torch.argsort(i, -1).unsqueeze(-1).expand(-1, -1, H))
    out = torch.zeros_like(per[:, 0])
    for k in range(K):
        out.add_(per[:, k])
    return h + out

mesh = init_device_mesh("cuda", (EP, world // EP), mesh_dim_names=("ep", "dp_shard"))
layout = build_moe_layout(num_experts=E, expert_parallel_size=EP, mesh=mesh, a2a_backend=BACKEND)
lo, n = layout.global_expert_offset, layout.local_num_experts
off = 0 if BACKEND == "deepep" else lo
def ep_layer(h, l):
    w, i = route(h, R[l])
    out = expert_parallel_forward(h, i, w, layout, lambda hh, ii, ww: grouped(hh, ii, ww, W1[l][lo:lo+n], W2[l][lo:lo+n], off))
    return h + out

def run(layer_fn):
    x = x0.clone().requires_grad_(True); h = x
    for l in range(L):
        h = checkpoint(layer_fn, h, l, use_reentrant=False) if CKPT else layer_fn(h, l)
    (h.float() * probe.float()).sum().backward()
    return h.detach(), x.grad

h_ref, g_ref = run(ref_layer)
h_ep, g_ep = run(ep_layer)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
res = torch.tensor([rel(h_ep, h_ref), rel(g_ep, g_ref), g_ep.float().norm().item() / g_ref.float().norm().item()], device=dev)
allr = [torch.zeros_like(res) for _ in range(world)]; dist.all_gather(allr, res)
if rank == 0:
    print(f"STACK EP={EP} {BACKEND} ckpt={CKPT}: per-rank out_rel/grad_rel/grad_ratio: " +
          " | ".join(f"r{i}:{a:.3f}/{b:.3f}/{c:.2f}" for i, (a, b, c) in enumerate(t.tolist() for t in allr)), flush=True)
dist.destroy_process_group()
