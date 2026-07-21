# 交叉苯偶姻 Δ-学习：项目实时状态

_持续更新的总览文档——不同于带日期的`REPORT_*`文件（某一时间点的快照），
本文件会随进展原地修改。配套文件：`STATUS_EN.md`。最后更新：2026-07-16
（2026-07-17新增勘误与第8轮相关章节）。_

## ⚠ 2026-07-17新增勘误：下文所有标†的冻结holdout MAE数字，都是在一个泄漏了骨架信息的切分上测出的——应视为偏乐观

本文档以及本项目历史上逐轮报告的每一个"冻结留出集MAE"数字（2.966 → 3.043 →
2.983 → 3.132 / 2.633 → 2.582 等，下文所有出现处都标了†），都是在
`candidates_v3`**最初**那版按分子级别（InChIKey不相交）划分的train/test切分上
测出来的。后续一次诊断发现，这个切分虽然分子不相交，却严重泄漏了
Bemis–Murcko**骨架(scaffold)**：**旧的29行冻结留出集里，93%的行至少有一侧
（供体或受体）的骨架已经出现在训练集中**（只有2/29对是真正意义上骨架全新的），
另外一个独立的、统计功效充足的骨架不相交CV测试（n=2255，重复5次）也发现了
一个真实、可复现的**+0.221 MAE（约9.8%相对）泛化差距**，出现在"内插"与"真正
全新骨架"之间。净效果：**本项目历史上报告的每一个冻结留出集数字，都把真实的
"面对全新化学结构"的泛化能力高估了大约0.2-0.5 MAE**——不要把它们当作真实
误差的紧上界来看待。完整细节见memory `cross-five-diagnostics-20260717`。

**这个问题已经向前修复。** `candidates_v3`被重建为一个真正骨架不相交的切分
（`rebuild_scaffold_disjoint_split.py`，对全部骨架做一次全局贪心、最大骨架
优先的装箱，已验证骨架零重叠），第1-7轮已标注的数据也按新切分重新打了标签
（干净训练集=19,687，**干净测试集=450**——相比旧的n=29，仅凭规模就是一个约
15.5倍更大、统计上更可信的留出集），两条模型架构线都在新切分上重新训练过。
诚实切分上的新headline数字：单模型XGB MAE 2.448/R² 0.730，MLP+XGB ensemble
MAE 2.256/R² 0.763，最佳GNN+ensemble blend MAE **2.215**（bootstrap已确认为
真实效应：P(blend优于ensemble)=0.9632，增益的90%置信区间为(-0.080, -0.003)，
B=20,000）。**这些新数字本身也有一个需要注意的前提**：如果直接拿它们和旧的†
数字比，看起来像是变好了，但那其实是一个成分构成上的假象——新的450行测试集
恰好偏向更容易的反应类型类别（偏杂原子-杂原子）。按代表性的类别比例重新加权
后，去掉泄漏之后诚实的、同类对比的代价是真实但幅度不大的**+0.14到+0.30
MAE**（相对旧的、仍然带泄漏的CV数字），与诊断阶段独立估计出的+0.221方向一致。
完整细节见memory `cross-scaffold-disjoint-rebuild-20260717`。

**今后请使用`data/cross_benzoin/cross_round7/
cross_train_table_7rounds_scaffold_split_labeled.parquet`（`new_scaffold_split`
列）作为评估资源，使用`cross_benzoin/predict_cross_champion.py`
（`CrossBenzoinBlendPredictor`）作为推荐的生产模型**——而不是下文提到的、
基于candidates_v3旧切分的冻结留出集/champion产出物（按本项目"保留历史"的
惯例予以保留；本勘误之前留下的每个数字，在每个小节第一次出现时都标了†）。

## 目前进展到哪里

**目标**：对任意一对醛(A+B，A可能等于B)，用Δ-学习廉价而准确地预测苯偶姻
缩合反应的ΔG（kcal/mol）：`dG_pred = dG_gxtb基线 + ML修正量`。目前存在
两条模型线：

| | 同源(A+A) | 交叉(A+B, A≠B) |
|---|---|---|
| 状态 | **已上线、已验证** | **完成6轮主动学习、已验证** |
| 标注覆盖 | 219,364个DFT标签（全部220,859个醛库） | 12,597个配对（25,176行有方向数据） |
| 最佳已验证模型文件 | `pipeline/models/gxtb_dft_correction_ENSEMBLE72_20260626.joblib`（测试MAE 1.503） | `data/cross_benzoin/cross_round6/train_ensemble_6rounds_slim120_v1/models/cross_ensemble_model.joblib`（冻结留出集MAE 2.582†，R² 0.897）——已被骨架不相交blend MAE 2.215取代，见上方勘误 |
| 推理入口 | `benzoin_dG.predict_dG_champion()` | 尚未打包成可安装的API，也尚未被晋升为未来主动学习轮次打分使用的模型（目前仍用单模型XGB champion这条线，见下文） |

**目前这是两条独立的模型线，不是一个统一的通用模型。** 合并已经测试过两次
（开放问题1），确实有帮助，但收益随交叉自身数据增长而萎缩——见下文。

## 交叉苯偶姻主动学习闭环：已完成6轮

冻结留出集始终是同一批29个分子不相交的`candidates_v3`测试行；这些行上的
g-xTB基线MAE（5.315）因此在每一轮都完全相同——这是一个持续有效的健全性
检查，说明轮次间的对比始终是公平的同类对比。**†下表所有"冻结集"行都是在
旧的、会泄漏骨架的切分上测出的——见本文档开头的勘误；诚实的骨架不相交
替代结果（n_test=450）是单模型XGB MAE 2.448/ensemble 2.256/blend 2.215。**

| | 第1-3轮 | 第1-4轮 | 第1-5轮 | 第1-6轮 |
|---|---:|---:|---:|---:|
| 行数/配对数 | 4,120/2,062 | 12,378/6,194 | 17,270/8,642 | 25,176/12,597 |
| CV champion MAE | 2.966 | 2.261 | 2.397 | 2.276 |
| CV champion R² | 0.750 | 0.751 | 0.409 | 0.473 |
| CV g-xTB基线MAE | 5.335 | 4.191 | 4.499 | 4.292 |
| 冻结集champion MAE † | 3.398 | 3.043 | 2.983 | 3.132 |
| 冻结集champion R² † | 0.842 | 0.859 | 0.865 | 0.854 |
| 冻结集ensemble MAE † | — | — | 2.633 | 2.582（该切分上目前最佳） |
| 冻结集ensemble R² † | — | — | 0.898 | 0.897 |

采样方法：第一轮=类别多样性；第二至六轮=bootstrap集成不确定性主动学习
（第六轮先针对第1-5轮champion对screen10k候选池剩余部分重新打分，再按
不确定性挑出前4,000/7,677对）。完整细节见带日期的报告，最新的是
`REPORT_cross_round6_completion_20260716_{EN,ZH}.md`。

**在当前方案下，`n_test`结构性地被封顶在29**：每一轮都只从
`candidates_v3`的`train`划分采样（这是正确的做法——新的训练轮次不应该
触碰留出的分子），所以测试集只有在未来某一轮**刻意**从`test`划分抽取
一些配对**仅用于评估、绝不用于训练**时才会增长。目前还没这么做；这值得
在某个时刻专门做一次决策，因为n=29会让报告的R²/MAE置信区间相当宽——下面
GNN-blend未能复现的发现就是一个具体的例子，说明这种谨慎态度确实有必要。

**除了单模型XGB champion之外，还有两条已经在冻结留出集上验证过的架构线**：
- **MLP+XGB ensemble**（`train_cross_ensemble.py`）：StandardScaler + MLP
  (128,64) + XGB(depth=5) + XGB(depth=7)，取平均。在测试过的每一个规模下
  都持续是最佳模型（第5轮：2.633†，第6轮：2.582†）——旧切分上目前最佳、
  已验证的产出物，但**尚未晋升**为`score_round_active_learning.py`给未来
  轮次打分使用的模型（目前仍用单模型XGB这条线）。第8轮准备工作应从下方
  "第8轮候选池准备"一节所述的池子里取数据。
- **三编码器GNN**（`train_cross_gnn.py`，同源预训练，GINE原子/键特征化+
  QM标量readout，对标同源项目已确认有效的双编码器方案）：在第5轮规模下，
  与ensemble做固定50/50 blend，以bootstrap显著性击败了单独的ensemble
  （P=0.987，n=29）。**这一结果在第6轮规模下未能复现**（P=0.456，不再
  显著；仅GNN甚至输给了ensemble，P(GNN更优)=0.096）——GNN是用完全相同、
  未经调参的超参数在约多46%的数据上重新训练的，绝对误差反而变差了
  （2.624→2.820），而ensemble持续改善。应该把这理解为"stacking是一个
  需要认真调参才能下结论的开放问题"，而不是"stacking不work"——一次
  未调参的重训练同样不是公平的测试。详见
  `REPORT_cross_round6_completion_20260716_{EN,ZH}.md`第3节。**后续更新
  （round6之后，先在当时仍带泄漏的旧切分上、后又在诚实的骨架不相交切分上
  重新确认）**：这个"未复现"结果其实是超参数没调好导致的假象——用
  `lr=3e-4, patience=25`之后blend稳定地优于ensemble（旧切分：P=0.9261，
  3个随机种子2.385±0.011；新的骨架不相交切分，n=450：P=0.9632，blend MAE
  2.215 vs ensemble 2.256，增益90%置信区间(-0.080,-0.003)）。见本文档开头
  的勘误以及memory `cross-scaffold-disjoint-rebuild-20260717`。

## 开放问题1 —— 把同源模型和交叉模型合并成一个通用模型？确认有帮助，但收益在萎缩

目前在两个规模下测试过，两次都是同样的260特征schema、同样的流程：

| 规模 | 仅交叉CV MAE | 合并后(仅交叉行)CV MAE | Δ | 相对改善 |
|---|---:|---:|---:|---:|
| 第1-3轮（4,120行） | 2.966 | 2.910 | -0.056 | -1.9% |
| 第1-5轮（17,270行） | 2.397 | 2.374 | -0.023 | -1.0% |

**解读**：合并的效果是真实的，方向上一直有帮助，但幅度随着交叉自身数据
增长4.2倍而大致减半——与"交叉数据越多，越不需要同源的助力"这一判断一致。
**第6轮（25,176行）是观察这个趋势是趋于平坦、继续萎缩、还是反转的自然
下一个数据点**——目前还没有在这个规模上重测。`assemble_cross_training_table_unified_v2.py`
已经可以泛化到任意轮数，所以重跑很便宜（零新增计算），只需要指向第6轮
的表即可。

## 开放问题2 —— 一个约10k规模的中等规模交叉产物计算是否值得？

已被后续进展取代：第4-6轮已经通过有针对性的主动学习新增了约21,000行
（不是盲目的10k规模扩张），这正是当初这个问题给出的、更有依据的路径。
最初观察到的学习曲线平台期（第1-3轮数据50%-100%时CV MAE持平在2.93-2.97）
在当时是成立的，但项目后续自身的历史（第4轮CV MAE降到2.261）说明，在
新的规模上做有针对性的主动学习，仍然可以让指标发生有意义的变化——在
某一轮数据内部观察到的平台期，不一定能预测下一轮的表现。

## 已知的小缺口（不紧急）

- `interaction_*`特征分组（10个特征）在测试过的每一轮（n=598到25,176）
  都确认没有用——可以在未来的清理中安全去掉，保留也无妨。
- 全库约0.15%的醛DFT-SP超时缺口（很早以前就记录过，见memory
  `dftsp-timeout-3600-snapshot`）每轮仍偶尔会损失几行；在当前规模下不
  值得专门回填。
- 产物BDE计算成功率在所有轮次中都稳定在约94%（部分环系的几何/成键判定
  失败）；可以接受，不需要处理。第6轮原始DFT-SP这一步本身是0/7,996个
  错误。

## 第8轮候选池准备（2026-07-17）——已备妥，等待用户批准再启动计算

第7轮已完成（见本文档开头的勘误）：数据集达到32,456行/16,241对，ensemble
CV MAE 1.883（目前最佳，在旧切分上）。screen10k候选池（第2-7轮不确定性
采样一直用的资源）现已完全耗尽（第5轮2,500 + 第6轮4,000 + 第7轮3,677 =
10,177/10,177全部用完）。

按骨架不相交重建工作的未完成事项第4条（memory
`cross-scaffold-disjoint-rebuild-20260717`）的要求，第8轮的候选池必须先
经过新的骨架切分查找表
（`data/cross_benzoin/candidates_v3/candidates_v3_pairs_with_scaffold_split.parquet`）
过滤，然后才能做不确定性打分——这样未来的轮次才不会像第1-7轮那样把
骨架泄漏进旧的分子级别冻结留出集里，而是保护好新的、诚实的骨架
test/validation集。

- **过滤后的可用池**：1,263,002对，同时满足(a)在`new_scaffold_split ==
  "train"`一侧（绝不是`test`/`validation`/`mixed`）和(b)尚未被第1-7轮
  标注（交叉核对：第1-7轮16,241对中有15,063对本身就在candidates_v3里、
  已被正确排除；另外1,178对来自screen10k等candidates_v3之外的来源，
  本表里自然不存在，符合预期）。
- **第8轮候选批次**：`cross_benzoin/sample_round8_from_candidates_v3.py`
  从这个可用池里做了按类别配平的采样（6个class_pair组合各667对=4,002个
  无序配对/8,004行有方向数据，随机种子42），批量大小与第6轮（迄今最大
  的一次定向新采样）相当。写入
  `data/cross_benzoin/cross_round8/cross_round8_pairs.csv`。**这一步
  没有用到任何计算资源**——只是对已有parquet/CSV表做pandas join+分层
  抽样，没有调用任何xTB/g-xTB/DFT。
- **尚未做、需要明确批准**：真正的不确定性*打分*（区别于上面的预选择）
  需要先对候选池做特征化（`cb_featurize.py`融合了xTB构象搜索+g-xTB单点
  计算——真实的计算量），`score_round_active_learning.py`再消费这些特征。
  本次会话刻意没有启动特征化。获批后按顺序执行的命令如下：

  1. **特征化**（xTB+g-xTB计算，genoa阵列作业，复用已缓存的220,859个醛
     库，因此只有产物侧是新计算）：
     ```
     IN=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_round8/cross_round8_pairs.csv
     OUT=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/cross_round8
     N=$(($(wc -l < "$IN")-1)); CHUNK=100; NCH=$(( (N+CHUNK-1)/CHUNK )); mkdir -p "$OUT/logs"
     sbatch --array=0-$((NCH-1))%64 --output="$OUT/logs/cb_%A_%a.out" \
       --export=ALL,INPUT="$IN",OUTDIR="$OUT",CHUNK=$CHUNK,\
     ALD_CACHE=/scratch-shared/schen3/benzoin-dg/data/cross_benzoin/homo_v6/aldehydes_all.csv,\
     REQUIRE_CACHE_COMPLETE=1 cross_benzoin/slurm/submit_cb_featurize_array.sh
     ```
  2. **合并各chunk输出**为一个`cross_round8_products.csv`，然后算**产物BDE**
     （`pipeline/compute/calc_bde_gxtb_product_cross.py --products-csv ... --out-dir
     data/cross_benzoin/cross_round8/bde_gxtb`，按chunk做阵列作业，与第2轮/
     第6轮的BDE后续作业配方相同）。
  3. **组装特征表**：`python cross_benzoin/assemble_cross_round_features.py --tag cross_round8`。
  4. **对修正后的、干净的骨架训练池做不确定性打分**（不是旧的、会泄漏的
     那个池子——参照第7轮骨架检查的先例，
     `slurm/submit_score_round7_scaffold_check.sh`）：
     ```
     python cross_benzoin/score_round_active_learning.py --tag cross_round8 \
         --train-table data/cross_benzoin/cross_round7/cross_train_table_7rounds_scaffold_clean_train.parquet \
         --feature-list data/cross_benzoin/cross_round7/scaffold_disjoint_v1/models/feature_list.json \
         --model ensemble --n-boot 40 --n-select 4000 --seed 42
     ```
  5. **DFT-SP**：对不确定性排名靠前的选择集，用
     `pipeline/compute/dft_sp_cross_from_geom.py`（r2SCAN-3c，与此前每一轮
     标注步骤配方相同）——只有在第1-4步完成且明确获批之后才执行。

## 第8轮计算已启动（2026-07-17，用户已批准——利用genoa空闲算力）

用户已批准启动第8轮流水线（批准时`sinfo -p genoa`确认84个空闲节点）。正在
按上文的5步顺序执行。

- **第1步（特征化）已提交**：作业 **24707568**，`cb_feat`阵列作业，
  `--array=0-80%64`（81个chunk × 每chunk 100对 = 8,004行），genoa分区，使用
  `ALD_CACHE=data/cross_benzoin/homo_v6/aldehydes_all.csv` 加
  `REQUIRE_CACHE_COMPLETE=1`，只对产物侧做新的xTB+g-xTB计算（供体/受体醛都
  已在缓存的220,859条全库中）。启动时已验证健康：20秒内20+个任务进入RUNNING
  状态，`chunk 0: pairs 0:100`切片正确，无立即报错。输出目录：
  `data/cross_benzoin/cross_round8/chunk_NNNN/`。该步骤历史上单个chunk耗时约
  1-2小时（偶尔约1分钟，当整个chunk命中缓存时），按%64节流、81个chunk估算，
  整个阵列作业预计约2-4小时跑完。
- 第2-5步（产物BDE、组装、打分、DFT单点）尚未开始，等待第1步完成。本节会
  随每一步完成持续更新。

## 下一步动作（截至2026-07-16，无先后顺序——尚未选定方向）

- **把第6轮的MLP+XGB ensemble晋升**为正式上线的champion / 
  `score_round_active_learning.py`给未来轮次打分使用的模型——它现在是
  每个测试过的规模下最佳、已验证的产出物。
- **对GNN做调参**（学习率/训练轮数/模型容量）之后再在第6轮规模上下
  结论，判断stacking是否值得继续投入——第5轮的"胜利"和第6轮的"失败"
  都只是各一次未调参的数据点。
- **第7轮**：第6轮用掉4,000对之后，screen10k候选池里还剩3,677/7,677对
  未使用——从中抽取零新增xTB/g-xTB计算。
- **第三次重测同源+交叉合并**，在第6轮（25,176行）规模上，使用已经
  泛化好的`assemble_cross_training_table_unified_v2.py`。

## 2026-07-20 补记：切分比例最终定案

用户指出配对级切分比例（60.7% train / 1.4% val / 1.5% test / 36.8% mixed，源于分子级
80/10/10目标经"两边都需同一split"规则平方压缩）看起来不合理，要求评估改为7:2:1。已完整
实测两个方案（分子级70/20/10 vs 80/10/10）的下游champion+ensemble+GNN重训练结果：

| | 80/10/10（production） | 70/20/10（存档对照） |
|---|---:|---:|
| clean train行数 | 19,687 | 14,874 |
| blend MAE（诚实holdout n=450） | **2.215** | 2.614 |
| bootstrap 90% CI（blend vs ensemble） | (-0.080, -0.003) 不跨零 | (-0.001, 0.100) 勉强跨零 |

用户最终决定：**80/10/10 定为production**（`scaffold_disjoint_v1`，
`predict_cross_champion.py`默认加载此版本），因为训练数据更多、精度更高、显著性检验更
干净，且7:2:1额外增大的validation桶只对GNN的early-stopping有用、对表格模型完全没用，
实测也未能体现出对应的精度优势。70/20/10版本（`scaffold_disjoint_721_v1`）完整保留作为
已验证的存档对照，不删除。round8起的所有候选池筛选、round9+的1.263M/960k可用池两个版本
都已生成（`candidates_v3_pairs_with_scaffold_split.parquet` = 80/10/10口径，
`_721.parquet` = 70/20/10口径），后续统一使用80/10/10口径的版本。

## 2026-07-20 补记：round8执行中发现标准5步流程缺了一步，已在脚本层面修复

实际跑round8时，`assemble_cross_round_features.py`拼表后发现缺53个`product_mordred_*`
特征（champion的260维特征schema从round6/7起就包含它们，但文档里的5步清单没写这一步）。
已修复：
1. **补齐当轮**：跑`add_mordred_cross_products.py`（复用已保存的xyz几何，无需重新算构象，
   ~1s/分子）产出mordred，重新拼表。
2. **修脚本本身，不只是补一次**：`assemble_cross_round_features.py`现在会自动探测
   `{rdir}/mordred_products/chunk_*.csv`；如果探测不到又没显式传`--product-mordred-csv`，
   直接**硬报错退出**（需要显式加`--allow-missing-mordred`才能继续），不会再像这次一样
   悄悄产出一张"看起来完整、实际缺53维特征"的表。

**round9+的正确6步流程**（不再是5步）：
1. 特征化（cb_featurize array，CHUNK=30，先跑`check_aldehyde_cache_coverage.py`预检）
2. 合并chunk + 产物侧g-xTB BDE array
3. **产物侧mordred array**（`add_mordred_cross_products.py`，新增的必须步骤）
4. 拼表（`assemble_cross_round_features.py`，现在会自动找mordred或硬报错）
5. 主动学习打分（`score_round_active_learning.py`，用production 80/10/10口径的模型）
6. DFT-SP标注

## 2026-07-20 补记:新champion——round1-8 + attentive pooling GNN,已确认显著优于旧版

round8的DFT-SP全部跑完(7,896/7,896,0错误)后,把round1-8合并成新训练表(40,352行,
清洁训练集从19,687增长到27,583),同时把之前排队已久的AttentiveFP式pooling架构改进
(3-seed确认稳健:GNN-only MAE 2.324 vs 默认架构2.563)应用到这一版重训练:

| | round1-7(旧champion) | **round1-8 + attentive(新champion)** |
|---|---:|---:|
| clean train行数 | 19,687 | 27,583 |
| ensemble-only MAE | 2.256 | 2.201 |
| **blend MAE(诚实holdout n=450)** | 2.215 | **2.106** |
| bootstrap 90% CI | (-0.080, -0.003) | (0.031, 0.159) |
| P(blend更好) | 0.9632 | **0.9923** |

`predict_cross_champion.py`已扩展支持attentive架构(从`gnn_norm_stats.joblib`的`arch`字段
自动识别模型类,兼容新旧两种checkpoint文件名),SLURM验证精确复现训练数字(2.1065 vs
报告的2.106)。**这是当前推荐的production champion**,产物路径:
`data/cross_benzoin/cross_round8/scaffold_disjoint_8rounds_v1`(表格champion+ensemble)+
`data/cross_benzoin/cross_round8/gnn_attentive_8rounds_v1`(GNN)。

过程中也修了一个真实bug:`train_cross_gnn_arch_sweep.py`原本是给架构对比搜索用的,没存
标准化参数(norm stats),导致第一次训练完的attentive模型没法直接封装——已补上保存逻辑
并重跑(镜像`train_cross_gnn_scaffold_disjoint.py`的存档模式)。
