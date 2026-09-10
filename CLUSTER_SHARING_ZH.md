# 集群资源分享规则 —— benzoin-dg ⇄ NHC-benzoin

> 两个项目（本 ΔG 项目 + NHC 催化/动力学项目）共用一个 Snellius 账号。规则的目的：
> 充分利用集群，但**永不占满一个分区** —— 更高优先级的作业（以及对方项目）必须始终
> 有空间。English: `CLUSTER_SHARING.md`（保持同步）。
>
> 2026-09-10 由 benzoin-dg 会话起草，待与 NHC 会话确认（写这份时 NHC 会话已重启，能
> 联系上就发提议）。**两个项目都适用。**

## 1. QOS 上限 —— 以及"远离上限"的规则

`MaxJobsPU = 128` 是**每分区**的：一个账号可以在 **rome 上跑到 128，同时 genoa 上跑到
128** → rome+genoa 合计上限 **256**。

**永远不要跑到上限。** 目标：

| 分区 | 合计（benzoin-dg + NHC）运行任务 | 留出余量 |
|---|---|---|
| **rome** | ≤ **~100** | ≥ ~28 |
| **genoa** | ≤ **~100** | ≥ ~28 |
| **rome + genoa 合计** | ≤ **~200**（不是 256） | ≥ ~56 |

- 在此之内，每个项目**自己**的稳态在任一分区 ≤ **~90**，且遇到以下情况再往下降：
  (a) 对方项目有作业在排队等资源；(b) 出现更高优先级 / 其他用户的作业。
- 短时冲高到目标以上**仅当**该分区其它方面空闲，且对方要用时降回。

## 2. 默认分区分配

| 分区 | 主用 | 说明 |
|---|---|---|
| **fat_genoa**（48 节点，4.3× 内存，贵 50%） | **NHC** | TS 搜索 / 微观动力学作业可能真需要内存。 |
| **rome**（521）+ **genoa**（737） | **benzoin-dg** | ORCA 单点 + xtb，~48 GB —— 放这里够。 |
| **fat_rome**（72 节点） | 共享，先到先得 | **仅**用于真正需要 > ~200 GB 的作业。 |
| GPU（gpu_h100 / gpu_a100） | 共享，先到先得 | —— |

## 3. fat 分区：用它，但别占满

- **fat 靠"内存需要"来用，不靠"其它分区拥挤"来用。** 作业若放得下 rome/genoa
  （≤ ~200 GB），rome/genoa 忙时就在那排队等，**不能**因为非-fat 分区一时占满就溢出
  到 fat。（这正是 benzoin-dg 2026-09-10 犯的错 —— 把内存很轻的 Tier B / homo-SP 阵列
  放到 fat_genoa。已纠正：移到 rome。）
- **占用一个 fat 分区永不超过 ~75%**：fat_genoa ≤ ~**36 / 48**，fat_rome ≤ ~**54 / 72**。
  给对方项目和更高优先级作业留 ≥ ~25%。

## 4. 操作机制

- `scontrol update JobId=<id> ArrayTaskThrottle=<n>` 降一个运行中阵列的并发；随运行中
  任务跑完而收敛。控制台 "permission denied" / "already finished" 是表面噪声，实际生效。
- 要*立刻*（不是 ~1 h 内收敛）腾资源：对一个**可断点续跑**的阵列
  `scancel --state=RUNNING <jobid>`（worker 逐对 append+fsync + 重启 pid-skip，所以每个
  被取消的任务最多丢 1 对）。Tier B 和 homo-SP 都可断点续跑。
- 提交会持有 **> ~80 并发任务**的阵列前：`squeue -u schen3 -h -o "%P %t" | sort |
  uniq -c` 看合计脚印；若对方有 PENDING 作业，先 ping 对方会话。

## 5. 解堵

任一方可（通过 SendMessage）请对方降一个 throttle。被请的一方照做并回复。两边是同一个
用户的工作 —— 合作。

## 6. 当前状态（2026-09-10，benzoin-dg 侧）

- **cross Tier B** —— 只在 rome，已降优先级（"Tier B 可以慢"）：臂 `26467946` %30 +
  `26554708` %25（合计 ~55 目标）。
- **homo relabel** —— 重投到 **genoa**，限到 benzoin-dg 的 genoa 数 ≤ ~70。
- benzoin-dg 在任何 fat 分区上都没有作业。
- NHC：`nhc-gsp-*` fat_genoa ~40（接近 75% 线 —— NHC 值得收一点），genoa ~5–25。
