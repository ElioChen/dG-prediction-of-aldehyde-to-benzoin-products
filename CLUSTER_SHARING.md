# Cluster sharing rules — benzoin-dg ⇄ NHC-benzoin

> Two projects (this ΔG project + the NHC catalyst / kinetics project) run under
> one Snellius account. Rules so we use the cluster well **without ever
> saturating a partition** — higher-priority jobs (and each other) must always
> have room. 中文版: `CLUSTER_SHARING_ZH.md` (keep in sync).
>
> Drafted 2026-09-10 by the benzoin-dg session; to be confirmed with the NHC
> session (it had restarted when this was written — proposal will be sent when
> it is reachable). **Applies to both projects.**

## 1. The QOS ceiling — and the rule to stay well under it

`MaxJobsPU = 128` is **per partition**: one account can run **up to 128 on rome
AND up to 128 on genoa** → a **256** combined ceiling across rome+genoa.

**Never run to the ceiling.** Targets:

| Partition | Combined (benzoin-dg + NHC) running tasks | Headroom left |
|---|---|---|
| **rome** | ≤ **~100** | ≥ ~28 |
| **genoa** | ≤ **~100** | ≥ ~28 |
| **rome + genoa together** | ≤ **~200** (not 256) | ≥ ~56 |

- Within that, each project's **own** steady-state ≤ **~90** on any single
  partition, and drops further when (a) the other project has jobs PENDING for
  capacity, or (b) higher-priority / other users' jobs appear.
- Short bursts above target are OK **only** when the partition is otherwise
  idle, and are throttled back on request.

## 2. Default partition assignment

| Partition | Primary | Notes |
|---|---|---|
| **fat_genoa** (48 nodes, 4.3× RAM, +50% billed) | **NHC** | TS-search / microkinetics jobs plausibly need the RAM. |
| **rome** (521) + **genoa** (737) | **benzoin-dg** | ORCA single-points + xtb, ~48 GB — fit here. |
| **fat_rome** (72 nodes) | shared, first-come | ONLY for jobs that genuinely need > ~200 GB. |
| GPU (gpu_h100 / gpu_a100) | shared, first-come | — |

## 3. fat partitions: use them, never fill them

- **fat is RAM-justified, not congestion-justified.** A job that fits on
  rome/genoa (≤ ~200 GB) queues there and waits when rome/genoa are busy — it
  does **not** spill onto fat just because rome/genoa are momentarily full.
  (This is the mistake benzoin-dg made 2026-09-10 — memory-light Tier B / homo-SP
  arrays on fat_genoa. Corrected: moved to rome.)
- **Never occupy more than ~75% of a fat partition**: fat_genoa ≤ ~**36 / 48**,
  fat_rome ≤ ~**54 / 72**. Leave ≥ ~25% for the other project and for
  higher-priority work.

## 4. Mechanics

- `scontrol update JobId=<id> ArrayTaskThrottle=<n>` lowers a running array's
  concurrency; it converges as running tasks finish. The "permission denied" /
  "already finished" console lines are cosmetic — it works.
- To free capacity *now* (not over ~1 h): `scancel --state=RUNNING <jobid>` on
  a **resume-safe** array (worker does per-pair append+fsync + pid-skip on
  restart, so ≤ 1 pair is lost per cancelled task). Tier B and homo-SP are both
  resume-safe.
- Before submitting an array that will hold **> ~80 concurrent tasks**:
  `squeue -u schen3 -h -o "%P %t" | sort | uniq -c` — check the combined
  footprint; if the other project has PENDING jobs, ping their session first.

## 5. Unblocking

Either side may ask the other (via SendMessage) to drop a throttle. The asked
side does it and replies. Same user's work on both sides — cooperate.

## 6. Current state (2026-09-10, benzoin-dg side)

- **cross Tier B** — rome only, deprioritized ("Tier B can be slow"): arms
  `26467946` %30 + `26554708` %25 (~55 combined target).
- **homo relabel** — relaunching on **genoa**, capped so benzoin-dg's genoa
  count stays ≤ ~70.
- Nothing of benzoin-dg's on any fat partition.
- NHC: `nhc-gsp-*` ~40 on fat_genoa (near the 75% line — worth NHC trimming),
  ~5–25 on genoa.
