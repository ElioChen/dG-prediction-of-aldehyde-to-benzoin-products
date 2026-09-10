# Cluster sharing rules — benzoin-dg ⇄ NHC-benzoin

> Two projects (this ΔG project + the NHC catalyst / kinetics project) run under
> one Snellius account and share the QOS. Proposed rules so neither starves the
> other. 中文版: `CLUSTER_SHARING_ZH.md` (keep in sync).
>
> Drafted 2026-09-10 by the benzoin-dg session; to be confirmed with the NHC
> session. **Not yet mutually agreed** (the NHC session had restarted when this
> was written — will send the proposal when it is reachable).

## 1. Default partition assignment

| Partition | Owner | Notes |
|---|---|---|
| **fat_genoa** (48 nodes, 4.3× RAM, +50% billed) | **NHC** | NHC's TS-search / microkinetics jobs plausibly need the RAM. benzoin-dg stays off it. |
| **rome** (521 nodes) + **genoa** (737 nodes) | **benzoin-dg** primary | ORCA single-points + xtb, ~48 GB — fit here. |
| **fat_rome** (72 nodes) | shared, first-come | ONLY for jobs that genuinely need > ~200 GB. |
| GPU (gpu_h100 / gpu_a100) | shared, first-come | — |

## 2. fat is RAM-justified, not congestion-justified

If a job fits on rome/genoa (≤ ~200 GB), it **queues there and waits** when
rome/genoa are busy. It does **not** spill onto a fat partition just because the
non-fat partitions are momentarily full. (This is the specific mistake
benzoin-dg made 2026-09-10 — putting memory-light Tier B / homo-SP arrays on
fat_genoa because rome/genoa were congested. Corrected: Tier B moved back to
rome.)

## 3. QOS MaxJobsPU = 128 (applies on rome and on genoa)

- Each project caps **steady-state** concurrency at **~90 per partition**, so the
  other always has ~40 headroom.
- Short bursts above ~90 are fine **only if the other partition is idle** and are
  dropped back when the other project needs it.
- `ArrayTaskThrottle` (`scontrol update JobId=<id> ArrayTaskThrottle=<n>`) is the
  tool — it converges as running tasks finish; the "permission denied" /
  "already finished" console lines are cosmetic, it works.

## 4. Courtesy before a big launch

Before submitting an array that will hold **> 100 concurrent tasks**:
- `squeue -u schen3 -h -o "%P %t" | sort | uniq -c` — check the combined footprint.
- If the other project has jobs PENDING for capacity, ping their session first.

## 5. Unblocking

Either side may ask the other (via SendMessage) to drop a throttle to clear a
block. The asked side does it and replies. This is the same user's work on both
sides — cooperate, don't compete.

## 6. Current state (2026-09-10)

- benzoin-dg: **cross Tier B** on rome only (arms `26467946` %50 + `26554708`
  %40, deprioritized — "Tier B can be slow"). **homo relabel** campaign will
  relaunch on **genoa**, capped ~90. Nothing of benzoin-dg's on any fat partition.
- NHC: `nhc-gsp-*` on fat_genoa + genoa.
