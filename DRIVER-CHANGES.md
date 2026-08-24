# Driver changes for gVisor GPU isolation

This fork of the open NVIDIA kernel modules adds **privileged mechanisms** that a
trusted host component drives to divide one GPU among mutually-untrusting gVisor
sandboxes. It is one layer of a three-project stack:

- **`vcluster-multitenant`** — cluster tenancy (who may run what).
- **`gvisor`** — the Sentry enforces each sandbox's GPU quota and compute share
  from userspace (the *policy*). See `gvisor/GPU-ISOLATION.md`.
- **`open-gpu-kernel-modules`** (this repo) — the *mechanism* that only kernel
  privilege can apply: driving the hardware runlist, and per-tenant UVM eviction.

**Why the driver must be involved (NVIDIA only).** After a channel is set up,
CUDA submits work by ringing a doorbell in mapped memory — that never enters the
kernel, so the Sentry cannot meter or preempt a doorbell-submission workload
(cuBLAS). Binding it requires manipulating the hardware runlist at kernel
privilege. The **mechanism** lives here; the **policy** that drives it lives in
the trusted host component `runsc gpu-scheduler` (never in the Sentry, which is
untrusted relative to the tenant). This is the same policy/mechanism split the
memory side uses. AMD needs no driver changes — its controls are `ioctl`s the
Sentry already interprets.

The changes are in two independent families, on two branches.

---

## 1. Ghost compute hooks — the runlist control surface

**Files:** `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c`,
`src/nvidia/src/kernel/gpu/fifo/kernel_ctxshare.c`.
**Applied via:** `gvisor/ghost-experiment/driver-hooks.patch` +
`driver-tsgs-report.patch` (the hooks are maintained as patches, not committed
into the vendored source). **Consumed by:** `runsc gpu-scheduler --runlist-control`.

Exposes `/proc/driver/nvidia/gpusched`, a root-only control surface with which
the trusted host scheduler observes and steers the hardware runlist:

- **`poll`** — dumps one line per tenant pid: `pid <p> active <0|1> tsgs <n>`.
  `active` is whether any of the tenant's channel groups is running; `tsgs` is
  how many channel groups (TSGs) it holds. The TSG count is what makes the credit
  scheduler's charge-back **per-tenant** rather than per-pid — without it a packed
  attacker (many forked processes = many TSGs behind one Sentry pid) is
  under-charged (`gvisor/SECURITY-FINDINGS.md` V4).
- **`attach` / `detach <chan>`** — remove or restore a channel group on the
  runlist, each followed by `NVA06F_CTRL_CMD_RESTART_RUNLIST` so GSP acts on the
  change. Detach evicts a whole tenant; the credit scheduler uses it to stop a
  tenant that has overdrawn its credit.
- **`ts <pid> <us>`** — set a tenant's per-TSG timeslice, again committed by
  `RESTART_RUNLIST`. Shifts a running tenant's share proportionally.

The companion `RESTART_RUNLIST` RPC is the non-obvious requirement: earlier
"temporal levers are inert" results were all missing it — `SET_TIMESLICE` and
`GPFIFO_SCHEDULE(disable)` act only on the *idle* runlist until a restart forces
GSP to re-read it. See `gvisor/NVIDIA-COMPUTE-ISOLATION.md` (CORRECTION 3) for
the full derivation.

`kernel_ctxshare.c` additionally carries the **spatial** TPC-partition probe
(`SET_TPC_PARTITION_TABLE` issued from the deferred `GPFIFO_SCHEDULE` site) and
the `GHOST_TOTAL_TPC` constant — half the GPU's SM count, a **per-GPU knob**
(A100 = 54, A6000 = 42, RTX 5070 = 24). That knob only affects the spatial
probes, not the credit scheduler.

---

## 2. UVM memory overcommit and per-tenant eviction

**Branch:** `gpu-overcommit-evict`. **Files:** `kernel-open/nvidia-uvm/`:
`uvm_ioctl.h`, `uvm_api.h`, `uvm.c` (the ioctl), `uvm_va_space.{c,h}`
(accounting), `uvm_pmm_gpu.{c,h}` (eviction). **Consumed by:** `nvproxy` on the
`gpu-overcommit` gVisor branch (`pkg/sentry/devices/nvproxy/{uvm,memquota}.go`).

### Why (measured)

Phase 1 (gVisor side) makes a tenant's GPU-memory *reservation* oversubscribable
(a `gmem` device-resident cap plus `hmem` host-swap headroom), relying on the
driver's native UVM eviction to keep residency within the device. But native
eviction is a **global LRU** that ignores per-tenant shares: a 4 GiB hot tenant
beside a 16 GiB oversubscriber on a 12 GiB card collapsed **281 → 13.5 GB/s**
(~20×) — the oversubscriber evicted the innocent tenant's resident pages
(`gvisor/OVERCOMMIT-SPIKE-FINDINGS.md`, S3). Overcommit is unsafe for
latency-sensitive tenants until eviction is per-tenant.

### The change

1. **Per-tenant residency accounting.** Each `uvm_va_space` (one per sandboxed
   process) and each *tenant group* (a Sentry-assigned id shared by all a
   sandbox's va_spaces — so forking cannot multiply protected residency, the
   memory analog of the compute packing attack) carries atomic `resident`,
   `limit`, and monotonic `evicted` byte counters, maintained at the 2 MB
   root-chunk granularity where the eviction lists are already updated
   (`gmem_account_chunk`, `uvm_gmem_add_resident`).
2. **Per-tenant victim selection.** `pick_root_chunk_to_evict` prefers a chunk
   whose tenant is **over its `gmem` cap** before falling back to the global
   oldest, so a tenant within its share is never the victim while any over-budget
   tenant has resident pages. A **background evictor kthread** proactively drains
   over-cap groups so the cap holds steadily under load.
3. **The `UVM_SET_GMEM_LIMIT` ioctl** (op 82, `uvm.c` route table). Sets a
   va_space's cap and tenant group and reads back the group's resident and
   evicted bytes. nvproxy issues it after `UVM_INITIALIZE` from the sandbox's
   quota, and re-issues it (idempotent) to sample the counters for **thrash
   detection** — a tenant pinned at its cap with a sustained eviction rate is
   paging on every access, the signal for a throttle/detach/kill policy.

The ioctl grows `UVM_SET_GMEM_LIMIT_PARAMS` over time (added `evictedBytes`);
UVM dispatches by raw index and copies `sizeof(params)`, so the struct can grow
without changing the op number, and an older Sentry simply ignores new fields.

### Status

Implemented and validated on the RTX 5070. With a 4 GiB cap, an innocent hot
tenant holds ~131 GB/s under a 16 GiB oversubscriber versus 13.5 GB/s under the
old global LRU (~10×); the driver evicts the over-budget tenant, not the
neighbour. Backward-compatible: with no cap set, the global LRU is used
unchanged. The residual drop from 281 to 131 GB/s is memory-bus contention from
the oversubscriber's paging DMA — a bandwidth-isolation problem residency does
not address (same class as CU masks not isolating VRAM bandwidth), and the
target of the thrash-response policy.

*(Considered and rejected: a broker-driven `UVM_MIGRATE` evictor. It has no
page-recency information, so it would migrate blindly and race the driver's
fault-time allocation. In-driver victim selection is surgical.)*

---

## Building and loading

The hooks are built and loaded by `gvisor/ghost-experiment/reload.sh` (compiles
`kernel-open/` and swaps the running modules) and reverted by `revert.sh` or a
reboot. Loading replaces the running driver, so schedule it when GPU workloads
are quiesced. The A100/A6000/5070 procedures are in
`gvisor/NVIDIA-COMPUTE-ISOLATION.md`.

## How this relates to the other projects

- **`gvisor`** drives everything here: `runsc gpu-scheduler` writes the runlist
  control surface (§1); `nvproxy` issues the UVM ioctl (§2). The Sentry holds the
  policy; this repo holds only mechanism. Start at `gvisor/GPU-ISOLATION.md`.
- **`vcluster-multitenant`** sits above gVisor and never touches the driver
  directly; it decides which tenants exist, which then map to sandboxes whose
  pids appear in the control surface above.
