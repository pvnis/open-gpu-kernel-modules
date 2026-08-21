# Phase 2 — per-tenant UVM residency eviction (design)

Branch `gpu-overcommit-evict` (off the working ghost driver). Nothing here is
loaded yet; the running driver is unchanged.

## Why (measured)

Phase 1 (in the gVisor tree) makes a tenant's GPU memory *reservation*
oversubscribable (gmem resident + hmem host-swap headroom), relying on the
driver's native UVM eviction to keep residency within the device. But native
eviction is a **global LRU**, so it does not respect per-tenant shares. Measured
(OVERCOMMIT-SPIKE-FINDINGS.md, S3): a 4 GiB hot tenant beside a 16 GiB
oversubscriber on a 12 GiB card collapsed from **281 GB/s to 13.5 GB/s (~20x)** —
the oversubscriber's working set evicted the innocent tenant's resident pages.
Overcommit is unsafe for latency-sensitive tenants until eviction is per-tenant.

## Where the global LRU lives (grounded)

`kernel-open/nvidia-uvm/uvm_pmm_gpu.c`:
- `pick_root_chunk_to_evict(pmm)` — chooses the victim. Prefers free chunks, then
  `get_first_allocated_chunk(pmm)`: the head of the **global**
  `pmm->root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED]`, i.e. the oldest
  allocated 2 MB root chunk across *all* va_spaces. This is the global LRU.
- `chunk_update_lists_locked(pmm, chunk)` (~:638) moves a root chunk onto the
  USED list when any subchunk is allocated — the residency/LRU-update hook.
- A chunk knows its owner: `uvm_gpu_chunk_t.va_block` (`uvm_pmm_gpu.h:293`), and
  `uvm_va_block_get_va_space(va_block)` (`uvm_va_block.h:698`) maps it to the
  `uvm_va_space_t`. Under gVisor each sandbox opens `/dev/nvidia-uvm` once, so
  **one va_space == one tenant.**

## The change

**1. Per-tenant residency state** — in `struct uvm_va_space_struct`
(`uvm_va_space.h:186`):
```c
atomic64_t resident_bytes;   // device-resident bytes owned by this va_space
NvU64      gmem_limit_bytes;  // device-resident cap; 0 = uncapped
```
Maintain `resident_bytes` at root-chunk granularity where the USED/eviction
lists are already updated (`chunk_update_lists_locked` and the free/evict paths,
`root_chunk_update_eviction_list`): add chunk size when a chunk becomes resident
for a va_space, subtract on free/evict. Root-chunk (2 MB) granularity matches the
eviction unit and avoids per-page counting.

**2. Per-tenant victim selection** — in `pick_root_chunk_to_evict`, before
falling back to the global oldest: prefer a chunk whose va_space is **over its
`gmem_limit_bytes`**. First implementation: scan the USED list for the first such
chunk (correct, simple, O(n) under pressure). Optimization (GVM's dual CLOCK):
keep a per-va_space list of resident root chunks and evict from an over-budget
tenant's own list, falling back to the global list only when the *whole device*
is oversubscribed and no single tenant is over cap. The scan version is the
milestone; the dual list is the follow-up.

Net effect: a tenant within its gmem share is never the victim while any
over-budget tenant has resident pages — A keeps its 4 GiB resident while B pages,
so A's 281 GB/s holds.

**3. Setting the cap** — the Sentry already knows each tenant's gmem
(`nvproxy.memAcct.gpuLimit`). Add a UVM ioctl `UVM_SET_GMEM_LIMIT` (a new op in
the ghost driver's `uvm.c` route table) that stores `gmem_limit_bytes` in the
va_space; nvproxy issues it once after `UVM_INITIALIZE` (add the struct + op to
`pkg/abi/nvgpu`, a handler in `pkg/sentry/devices/nvproxy/uvm.go`, and drive it
from `Register` using `gpuLimit`). This mirrors how the compute side plumbs
per-tenant policy to the driver, keeping policy in the Sentry and mechanism in
the driver.

## Approach decision

Considered a **broker-driven** alternative: expose per-va_space residency, and
have a trusted host component issue `UVM_MIGRATE(range → CPU)` to evict an
over-budget tenant's pages. Rejected as the primary path: the broker has no
page-access-recency information, so it would migrate blindly and thrash, and it
races the driver's own fault-time allocation. In-driver victim selection is
surgical (evicts the actual cold pages at the moment of pressure) and is exactly
GVM's design. Keep migration only as a possible coarse fallback.

## Validation plan

Re-run S3 with the cap set (A gmem=4 GiB, B gmem=4 GiB + hmem): A's bandwidth must
hold near 281 GB/s while B oversubscribes, instead of collapsing to 13.5. Then
the two-tenant vLLM overcommit case. Build with the driver's `reload.sh`; the
working ghost driver is restored by `revert.sh` / reboot. Note loading this
driver replaces the running one, so schedule it when the slicing workloads are
quiesced.

## Status

Design only. Next: implement (1) accounting + (3) the ioctl first (makes
per-tenant residency observable and settable, independently useful), then (2)
the scan-based victim selection, measure against S3, then the dual-list
optimization.
