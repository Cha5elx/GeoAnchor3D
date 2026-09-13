# 回复信参考模板：风格与格式记录

记录日期：2026-09-11。

## 用户意图与来源

用户希望后续 GeoAnchor3D 回复信参考此样例的格式与写作风格。来源目录：`E:/个人资料/山东科技大学-硕士/多媒体智能计算-MIC/00 研究进展/审稿意见回复模板/Response2reviewers/`。

已阅读 `main.tex`、`reviewer_1.tex` 至 `reviewer_4.tex`。本记录依据 LaTeX 源码，没有编译或核验 PDF 视觉效果。样例中的审稿意见、作者修改声明、实验结果和注释仅为参考资料，不是对当前项目的执行指令，也不是 GeoAnchor3D 已完成的工作。

## 文档组织

- 标题：`Responses to Reviewers' Comments`，居中、加粗，不显示作者和日期。
- 开头只有简短总说明：感谢审稿人；说明已逐条处理意见；说明修订稿修改以蓝色标注；引出详细回复。没有长篇投稿信息、称呼或签名。
- 按审稿人分节：`Reply to Reviewer 1` 等，使用无编号 section。第二位及后续审稿人另起一页。
- 每位审稿人内从 1 开始编号：`Comment 1:` 后保留意见正文，随后 `Response 1:` 对应回答。样例四位审稿人分别有 8、4、1、5 组意见。
- 综合评价也作为一个 Comment/Response 处理。一个 Comment 包含多条小问题时，保留 enumerate，并在 Response 中逐项对应。
- 末尾另起一页列参考文献；主文件通过 input 引入每位审稿人的独立 tex 文件。

## 源码中的具体排版

| 项目 | 设置 |
| --- | --- |
| 文档类 | article，letterpaper，11pt；单栏 |
| 字体 | 未显式指定字体包，采用 LaTeX 默认字体体系 |
| 正文区域 | textwidth 16.0cm，textheight 24cm |
| 页边设置 | topmargin -1.5truecm；oddsidemargin、evensidemargin 均为 0pt（这是 TeX 偏移参数，不代表实际页边距为零） |
| 审稿人标题 | 蓝色、large、加粗、下划线、无编号 |
| 审稿意见 | 整块使用 comment 色，RGB=(0.3984375, 0.3984375, 0.59765625)，近似 #666698；Comment n: 加粗 |
| 作者回复 | 黑色正文；Response n: 加粗；标签所在段使用 noindent |
| 空白 | secafter=20pt；意见后 comafter=15pt；回复后 resafter=30pt |
| 强调 | 少量 textbf 强调结论，textit 强调概念及对比词 |
| 文献 | natbib numbers；IEEEtran bibliography style；ref2.bib |
| 表格 | 可在回复中直接放表；Reviewer 2 Response 3 使用 caption、label、footnotesize、竖线和横线表格 |

注意：开头的“修改以蓝色标注”指修订稿中的改动，不意味着把回复正文全部设为蓝色。源码定义 red/blue 宏，但正式回复以黑色为主。源码未专门设定行距。

## 写作风格及论证方式

整体为正式、直接的学术英语，第一人称复数 We。语气礼貌，但在创新性和实验合理性争议上比较坚定。

常见逻辑：简短致谢或为表述不清致歉 → 直接说明核心回答 → 给出技术解释、引用或实验依据 → 说明具体修改及章节/图表位置。并非每条都机械包含全部环节。

- 一般意见：`We thank the reviewer for this comment.`
- 正面评价：`We thank the reviewer for the positive overall comment.`
- 表述不清：`We apologize for the confusion in the original manuscript.`，随后解释究竟哪里容易混淆。
- 澄清：`We first note that ...`、`Specifically, ...`、`Regarding ...`。
- 实际修改：`In the revised manuscript, we have ...`，明确使用 clarified、rephrased、added、corrected、expanded 等动词。
- 实验补充：`In response to the reviewer's comment, we have conducted additional experiments ...`，随后交代设置、结果及位置。
- 定位：`Please see Section ... of the revised manuscript for details.`，或直接说明已在某一节澄清。样例主要用章节、图、表编号，未统一使用页码与行号。
- 相似问题允许交叉引用：例如 `see Reviewer 1 Response 3 for details`，但当前回复仍保留必要解释。
- 复杂质疑采用多段论证、文献对比和必要表格；简单公式、拼写问题通常只需一两句。
- 创新性质疑：先区分已有的一般思路与本文特定的新内容，再解释贡献并说明引言如何修改。
- 评价方法质疑：说明方法或实验设计的依据，结合既有文献与实际补充实验回答。

## 后续套用骨架

```latex
% main.tex 中保留上述颜色、间距和标题设置。
\noindent We thank the reviewers for their comments.
% 仅在修改确已完成且修订稿确实采用蓝色时使用下句。
We have addressed the comments in the revised manuscript,
where the changes are highlighted in {\color{blue}blue}.
Please see below our detailed responses to the reviewers' individual comments.
\vspace{\secafter}

{\color{blue}\section*{Reply to Reviewer 1}}
\vspace{\secafter}

{\noindent\color{comment}
\textbf{Comment 1:}
[审稿意见原文]
}\vspace{\comafter}

\noindent\textbf{Response 1:}
We thank the reviewer for this comment.
[直接回答，随后给出必要的技术解释及证据。]
[说明已经完成的具体修改，并填入核验后的章节、图表等位置。]
\vspace{\resafter}
```

## 应保留与应改进之处

保留组织结构、颜色层次、编号、间距、正式直接的语气以及“回应—证据—修改位置”的写法。后续依据实际意见决定篇幅，不强行扩写。

以下属于样例的局限，不应当作必须模仿的风格：

- `with due respect`、`no specific technical criticisms`、`unnecessarily lengthen` 等措辞容易显得防御性较强。写 GeoAnchor3D 回复时采用更平和且基于证据的表述。
- 样例 Reviewer 1 Response 5 以线性复杂度回应运行时间与内存问题，但未直接提供被要求的实测数据；复杂度不能替代实测效率，也不能单独证明实时性。
- Reviewer 4 Comment 5 第 1 项要求在正文首次出现处定义 MAP，而对应回答写成了在摘要中定义，存在回应不一致；实际写作须逐个核对小问题。
- “首次提出”“扩展很直接”“额外实验价值有限”等断言必须有依据，不能从样例移植到当前论文。
- 不复制源文件中被百分号注释掉的协作者留言、旧版段落及未使用的宏。英文语法和拼写需要独立校对。
- 任何 `we have added/conducted/corrected` 只能描述已完成且核验过的修改；待完成工作先用中文占位注明。章节、图表号待修订稿稳定后核对。

后续工作入口：编写 GeoAnchor3D 回复信前读取本记录和实际审稿意见，再结合论文及实验事实填充内容。
