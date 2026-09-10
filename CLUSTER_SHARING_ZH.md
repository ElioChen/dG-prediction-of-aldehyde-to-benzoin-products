# 集群资源分享规则 —— benzoin-dg ⇄ NHC-benzoin

> 两个项目（本 ΔG 项目 + NHC 催化/动力学项目）共用一个 Snellius 账号、共享 QOS。
> 下面是提议的规则，避免互相饿死。English: `CLUSTER_SHARING.md`（保持同步）。
>
> 2026-09-10 由 benzoin-dg 会话起草，待与 NHC 会话确认。**尚未双方达成一致**
> （写这份时 NHC 会话已重启，能联系上就把提议发过去）。

## 1. 默认分区分配

| 分区 | 归属 | 说明 |
|---|---|---|
| **fat_genoa**（48 节点，4.3× 内存，贵 50%） | **NHC** | NHC 的 TS 搜索 / 微观动力学作业可能真的需要内存。benzoin-dg 不用它。 |
| **rome**（521 节点）+ **genoa**（737 节点） | **benzoin-dg** 主用 | ORCA 单点 + xtb，~48 GB —— 放这里够。 |
| **fat_rome**（72 节点） | 共享，先到先得 | **仅**用于真正需要 > ~200 GB 的作业。 |
| GPU（gpu_h100 / gpu_a100） | 共享，先到先得 | —— |

## 2. fat 分区靠"内存需要"来用，不靠"其它分区拥挤"来用

作业若放得下 rome/genoa（≤ ~200 GB），rome/genoa 忙时就**在那排队等**，
**不能**因为非-fat 分区一时占满就溢出到 fat。（这正是 benzoin-dg 2026-09-10 犯的错
—— 把内存很轻的 Tier B / homo-SP 阵列放到 fat_genoa，理由只是 rome/genoa 拥挤。
已纠正：Tier B 移回 rome。）

## 3. QOS MaxJobsPU = 128（rome 和 genoa 各有）

- 每个项目**稳态**并发上限 **~90 / 分区**，给对方留 ~40 余量。
- 短时冲高到 ~90 以上**仅当对方分区空闲**，对方要用时降回。
- 工具 = `scontrol update JobId=<id> ArrayTaskThrottle=<n>`，随运行中任务跑完而收敛；
  控制台 "permission denied" / "already finished" 是表面噪声，实际生效。

## 4. 大批量提交前打招呼

提交会持有 **> 100 并发任务**的阵列前：
- `squeue -u schen3 -h -o "%P %t" | sort | uniq -c` —— 看合计脚印。
- 若对方有作业在排队等资源，先 ping 对方会话。

## 5. 解堵

任一方可（通过 SendMessage）请对方降一个 throttle 来解堵。被请的一方照做并回复。
两边是同一个用户的工作 —— 合作，不竞争。

## 6. 当前状态（2026-09-10）

- benzoin-dg：**cross Tier B** 只在 rome（臂 `26467946` %50 + `26554708` %40，已降优
  先级 —— "Tier B 可以慢"）。**homo relabel** 战役重投到 **genoa**，上限 ~90。
  benzoin-dg 在任何 fat 分区上都没有作业。
- NHC：`nhc-gsp-*` 在 fat_genoa + genoa。
