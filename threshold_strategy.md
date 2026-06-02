# 猫识别阈值策略说明

## 1. 目标

当前识别系统面向普通用户时，用户很可能不认识校园猫。因此阈值策略的核心目标不是让系统尽可能多地自动确认，而是：

- `confirmed`：尽量少误认，直接展示猫档案。
- `uncertain`：给出 Top3 候选，但不要求普通用户确认，可进入后台审核。
- `unknown`：表示可能是未入档新猫、图片质量差，或系统无法可靠判断。

## 2. 当前识别流程

```text
上传图片
↓
YOLO 裁出猫
↓
DINOv3 提取图像向量
↓
FAISS 检索 Top3 相似候选
↓
根据 top1_score 和 gap 判断 confirmed / uncertain / unknown
```

其中：

- `top1_score`：第一候选和查询图的向量相似度。
- `top2_score`：第二候选和查询图的向量相似度。
- `gap = top1_score - top2_score`：第一候选领先第二候选的幅度。

`score` 是向量余弦相似度，不是概率。`0.80` 不代表 80% 概率正确。

## 3. 为什么要使用 gap

只看 `top1_score` 不够可靠。测试集中存在错误预测也有较高 score 的情况。

基于 `outputs/split_baseline/predictions.csv` 的统计：

```text
测试集数量：578
Top1 accuracy：82.35%
Top3 accuracy：90.31%

Top1 正确样本：
  平均 top1_score ≈ 0.787
  median gap ≈ 0.126

Top1 错误样本：
  平均 top1_score ≈ 0.650
  median gap ≈ 0.022
```

这说明错误预测通常有一个特点：第一名和第二名分数很接近。因此 confirmed 必须同时满足：

```text
top1_score 足够高
gap 足够大
```

## 4. 推荐阈值

当前采用平衡方案：

```python
CONFIRMED_THRESHOLD = 0.66
GAP_THRESHOLD = 0.07
UNCERTAIN_THRESHOLD = 0.55
```

决策规则：

```python
if top1_score >= 0.66 and gap >= 0.07:
    status = "confirmed"
elif top1_score >= 0.55:
    status = "uncertain"
else:
    status = "unknown"
```

## 5. 当前测试集效果

在 8:2 gallery/test split 的测试结果上，该策略表现为：

```text
confirmed 数量：328 / 578
confirmed 覆盖率：56.75%
confirmed 正确数：322
confirmed precision：98.17%

uncertain 数量：223
uncertain Top3 命中率：81.61%

unknown 数量：27
unknown 比例：4.67%
```

解释：

- 超过一半的测试图片可以自动展示猫档案。
- 自动 confirmed 的误认率约为 `6 / 328 = 1.83%`。
- uncertain 结果适合展示候选档案，并进入后台审核。

## 6. 产品展示建议

### confirmed

展示文案可以更确定：

```text
这是：猫名
```

直接展示猫档案、照片、性格、常出没地点。

### uncertain

不要让普通用户承担最终判断。展示文案建议更谨慎：

```text
它可能是这些猫
```

展示 Top3 候选档案卡片，并把这次识别记录进入后台待审核。

### unknown

展示文案：

```text
可能是未入档的新朋友，或图片暂时无法可靠识别
```

允许用户提交照片、地点、时间和备注，交给猫协审核。

## 7. 更保守的备选方案

如果演示或上线阶段更怕误认，可以切换为保守方案：

```python
CONFIRMED_THRESHOLD = 0.80
GAP_THRESHOLD = 0.12
UNCERTAIN_THRESHOLD = 0.55
```

该方案在当前测试集上的表现：

```text
confirmed precision：98.43%
confirmed 覆盖率：33.04%
wrong confirmed：3
uncertain Top3 命中率：87.78%
```

代价是更多结果会进入 uncertain。

## 8. 后续注意事项

当前阈值是基于已知猫的 gallery/test split 调出来的，还没有专门使用 unknown 负样本校准。

后续建议补充：

- 未入库猫图片。
- 非猫图片。
- 模糊图、远景图、遮挡图。
- 多猫同框图片。

这些样本可以用来更准确地校准 `unknown` 阈值，避免系统把库外猫强行识别成库内猫。

## 9. 当前代码位置

阈值常量和判断逻辑位于：

```text
services/identify_service.py
```

当前线上判定逻辑已经使用：

```text
confirmed_threshold = 0.66
gap_threshold = 0.07
uncertain_threshold = 0.55
```
