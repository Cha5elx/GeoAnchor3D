# 本轮回复调整与论文同步建议

依据用户新确认的信息：旧门控消融在 ScanRefer、ScanQA 两个数据集上训练；新增消融在四个数据集上联合训练；效率比较使用相同输入与推理配置。多随机种子结果尚未提供，继续保留待补状态。

已同步更新三个英文文件及对应中文文件中的九条回复：R1.1、R1.2、R1.4、R2.3、R3.2、R3.3、R3.8、R3.9、R3.10。R2.1 按此前要求保持不变。未修改论文文件。

## 1. 两组消融不是同一训练设置

训练数据不同可以解释为何两个表不能直接比较，不应将结果差异误认作同设置的不一致；也不能把分数的全部变化定量归因于增加训练数据，因为当前没有进一步的隔离实验。

回复已明确区别。建议在论文原门控消融或新消融段落附近加入：

```latex
The original gating-design ablation is trained on ScanRefer and
ScanQA only, whereas the new gating-granularity and supervision
ablation jointly trains all variants on ScanRefer, Multi3DRefer,
ScanQA, and Scan2Cap. The scores in these two tables therefore
correspond to different training settings; comparisons should be
made between variants within each setting.
```

如果层选择及聚合策略表也遵循同一个两数据集训练方案，可再将第一句的适用范围明确扩展到这两个表。不要把这句笼统扩展到所有旧表，尤其是含四数据集结果的组件消融表。

## 2. 相同条件下的效率实测可以直接报告

此前对比较条件是否一致的疑问，现已得到用户确认。当前回复已写明相同输入和推理配置，不再将设置差异作为潜在加速解释。

测得更快是实验观察，不需要自行否定；但这不等于 IGGA 在任何实现和硬件上都必然加速。无需在回复里猜测原因，也不应在无分析时归因于更稀疏的注意力或减少计算。

可在论文效率段增加一句：

```latex
Both models are evaluated with the same inputs and inference
configuration. The lower latency is reported as an observation
under this controlled benchmark, rather than an inherent
acceleration property of IGGA.
```

预热次数、重复次数等具体数值仍未提供，因此未自行添加。无需为了回答本轮问题重新运行实验；若已有记录，可用于完善复现说明。

## 3. 几何证据：分清两项实验回答的问题

- 组件消融回答：引入 GATH 对任务性能是否有帮助。
- 逐层线性探测回答：完整 GeoAnchor3D 的隐藏表示中，坐标是否更容易被线性恢复。
- t-SNE 回答：可视化中的局部聚类是否与物理坐标呈现定性关联。

三者可以形成互补证据，但不应合并为“已单独证明 GATH 保持了全局拓扑”。较合理的主张是：**辅助几何监督有任务收益，完整框架在深层保留了更多可线性恢复的坐标信息。**

以这一范围回应，现有实验已有实质支撑，不必为了维持更强的原措辞而立即增加实验。只有坚持“GATH 单独导致拓扑保持”或严格证明遗忘机制时，才需要相应的隔离对照与指标。

建议替换论文最后一段 t-SNE 讨论，保留实验描述并收窄结论：

```latex
To visualize the geometry of the intermediate representations,
we apply t-SNE to $\mathbf{h}_{\text{geo}}^{(i)}$ and color the
embeddings by physical X-axis coordinate. As shown in
Fig.~\ref{fig:GATH_tsne}, objects with similar X-coordinates tend
to form neighboring clusters after controlling for semantic
category. This pattern provides qualitative evidence of an
association with physical coordinates. The linear probes in
Table~\ref{tab:layerwise_probe} provide quantitative evidence of
coordinate recoverability in the complete GeoAnchor3D framework,
while the component ablation in Table~\ref{tab:final_ablation}
separately evaluates GATH's contribution to task performance.
```

这里不声称已测量全局距离保持、不声称单独分离 GATH 的探测收益，也不声称坐标信息随层数严格单调下降。原始可视化中的“控制语义类别”描述沿用当前论文。

## 4. 未开展的对照：以研究目标和控制变量解释

不建议使用“其他基于 Chat-Scene 的论文也没做”，因为其他论文的做法不能直接回答当前审稿人的要求。本轮也未检索或验证这一文献概括。

更合适的逻辑：明确当前问题 → 解释固定基础组件的目的 → 说明现有实验实际支持什么 → 简短承认未覆盖的维度。

**候选质量和主干：** 固定 Mask3D 与 Vicuna，是为了评估给定物体中心框架内几何调控的收益，避免把收益与检测器或 LLM 的变化混淆。实验已经使用实际预测的候选，并非只在理想物体上测试。但这只支持该输入设置，不证明对所有检测器质量或模型规模都鲁棒。

**手工几何编码：** 它是 IGGA 自身的设计选择，不能完全由“沿用 Chat-Scene”解释。回复应强调显式五维几何先验的紧凑性和可解释性，以及几何相容性映射仍然可学习。本文考察的是如何按指令、按头调控几何作用，而非哪种空间编码器最优。没有对照就不声称手工编码优于可学习嵌入。

这些理由比单独写“超出范围、留作未来工作”更具体，也比承诺并未完成的实验更可信；仍不能保证审稿人放弃其原实验要求。

## 5. 当前尚待完成

- R1.5、R3.5：等待真实多随机种子结果；超参数选择依据及敏感性分析仍需按实际完成情况填写。
- R3.10：回复已调整为现有证据可以支撑的范围，论文可视化段尚待采用上述或等价措辞。
- 训练设置和相同效率条件已经写入回复，但论文尚待同步。
- R2.1 依用户要求保留原回答，本轮未改公式或相关回复。
