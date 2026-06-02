# 复旦校园猫猫数字档案系统 · 项目方案

> 用AI让每只校园猫被看见、被记住
> 复旦大学 · 5人团队 · 1周开发周期

---

## 一、项目概述

### 定位

校园猫猫数字档案与偶遇社区。爱猫协会用它管理，普通同学用它偶遇，猫猫用它被记住。

### 核心功能

- 上传照片 → AI识别猫猫身份 → 展示完整档案
- 每只猫的数字画像：照片、性格故事、偶遇记录、出没地图
- 双层用户体系：猫协志愿者（管理后台）+ 普通同学（偶遇社区）

### 技术亮点

- YOLO + DINOv3 + FAISS 多模态识别Pipeline
- 置信度分级的优雅降级设计
- LLM自动生成猫猫性格故事
- 云端+本地混合架构（可选，拿附加分）

---

## 二、技术架构

### 系统分层

```
用户层     猫协志愿者            普通同学
              ↓                    ↓
输入层     照片上传 + 文字记录     偶遇照片上传
              ↓                    ↓
AI层              猫猫身份识别
              YOLO裁图 → DINOv3特征 → FAISS检索
              ↓
处理层     特征入库 / 位置记录 / 文字标签提取
              ↓
数据层          猫猫数字档案数据库
              ↓
展示层     档案页 / 偶遇动态流 / 热力地图
```

### 数据库结构

```sql
-- 猫猫主表
CREATE TABLE cats (
    id              VARCHAR(20) PRIMARY KEY,  -- 如 cat_001
    name            VARCHAR(50) NOT NULL,
    nickname        VARCHAR(100),
    gender          VARCHAR(10),
    neutered        BOOLEAN DEFAULT FALSE,
    age_estimate    VARCHAR(20),
    personality     TEXT,                     -- 性格标签，逗号分隔
    story           TEXT,                     -- AI生成的性格故事
    primary_photo   VARCHAR(200),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 猫猫照片表（用于识别入库）
CREATE TABLE cat_photos (
    id              SERIAL PRIMARY KEY,
    cat_id          VARCHAR(20) REFERENCES cats(id),
    photo_path      VARCHAR(200),
    feature_vector  BYTEA,                    -- DINOv3特征向量序列化
    angle           VARCHAR(20),              -- 正面/侧面/俯视
    lighting        VARCHAR(20),              -- 白天/阴天/夜晚
    is_primary      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 偶遇记录表
CREATE TABLE sightings (
    id              SERIAL PRIMARY KEY,
    cat_id          VARCHAR(20) REFERENCES cats(id),
    photo_path      VARCHAR(200),
    location_name   VARCHAR(100),             -- 如"图书馆门口"
    latitude        FLOAT,
    longitude       FLOAT,
    confidence      FLOAT,                    -- 识别置信度
    spotted_by      VARCHAR(50),              -- 用户标识
    note            TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

### 核心识别Pipeline

```python
# Step 1: YOLO裁出猫体
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')

def crop_cat(image):
  results = yolo(image)
  for box in results[0].boxes:
    if int(box.cls) == 15:  # COCO类别15=cat
      x1, y1, x2, y2 = box.xyxy[0]
      return image.crop((x1, y1, x2, y2))
  return image

# Step 2: DINOv3 (timm) 提取特征
import timm
import torch
import torchvision.transforms as T
model = timm.create_model('vit_base_patch16_dinov3', pretrained=True, num_classes=0, global_pool='token')
preprocess = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

def extract_features(image):
  tensor = preprocess(image).unsqueeze(0)
  with torch.no_grad():
    feats = model.forward_features(tensor)
    if feats.ndim == 3:
      feats = feats[:, 0, :] if feats.shape[1] > 1 else feats.mean(dim=1)
    feats = feats / feats.norm(dim=-1, keepdim=True)
  return feats.detach().cpu().numpy()

# Step 3: FAISS检索
import faiss
import numpy as np
# index should be constructed with the vector dimensionality
# index = faiss.IndexFlatIP(feature_dim)

def search_cat(query_features, top_k=3):
  scores, indices = index.search(query_features, top_k)
  return scores[0], indices[0]

# Step 4: 置信度分级
def classify_result(top_score):
  if top_score > 0.80:
    return "confirmed"
  elif top_score > 0.50:
    return "uncertain"
  else:
    return "unknown"
```

### API接口定义

```
POST /identify
  输入: 图片文件 (multipart/form-data)
  输出: {
    cat_id: "cat_003",
    cat_name: "橘总",
    confidence: 0.92,
    status: "confirmed" | "uncertain" | "unknown",
    candidates: [{cat_id, cat_name, confidence}, ...],  // uncertain时返回Top3
  }

GET  /cats
  输出: [{id, name, primary_photo, personality, last_seen}, ...]

GET  /cats/{id}
  输出: {id, name, story, photos[], sightings[], personality_tags[]}

POST /sightings
  输入: {cat_id, location_name, latitude, longitude, photo_path, note}
  输出: {sighting_id, created_at}

GET  /sightings?limit=20
  输出: [{cat_name, cat_photo, location_name, spotted_at, note}, ...]

GET  /map/heatmap
  输出: [{cat_id, cat_name, latitude, longitude, count}, ...]
```

---

## 三、5人分工

| 角色 | 任务 | 核心产出 |
|------|------|---------|
| **AI-1** | YOLO裁图 + DINOv3特征提取 + FAISS检索 | 识别Pipeline跑通 |
| **AI-2** | 数据清洗入库 + 准确率调优 + 边界case | 10只猫全部入库，准确率>80% |
| **后端** | FastAPI接口 + 数据库 + LLM故事生成 | 所有API可调用 |
| **前端-1** | 上传页 + 识别结果页 + 猫猫档案页 | 主流程跑通 |
| **前端-2** | 偶遇动态流 + 校园地图 + 整体视觉 | 完整UI可演示 |

**数据收集**：全员第一天分摊，每人负责联系猫协拿2-3只猫的照片和信息。

**答辩准备**：全员第六天参与，不单独占坑。

---

## 四、七天开发计划

### Day 1 — 全员对齐（最重要）

**上午：开1小时对齐会**

产出物：
- 数据库表结构（最终版，写入共享文档）
- API接口契约（最终版，写入共享文档）
- 猫协联系人确认，照片收集分工

**下午：各自行动**

| 角色 | 任务 |
|------|------|
| AI-1 | 装环境：YOLO、DINOv3(timm)、FAISS权重全部下载到本地 |
| AI-2 | 整理猫协照片，建立标注表格（猫名 + 照片路径） |
| 后端 | 搭FastAPI骨架，所有接口写mock，返回假数据跑通 |
| 前端-1 | 搭React项目，上传页静态UI做出来 |
| 前端-2 | 档案页 + 动态流静态UI做出来 |

**今天结束验收**：后端mock接口全部可调用，前端能跑起来。

---

### Day 2 — 各自并行开发

**AI-1：跑通3只猫的识别**
- YOLO检测 + DINOv3特征提取完成
- 3只猫照片入库FAISS
- 封装`/identify`接口（哪怕只能识别3只猫）

**AI-2：数据整理**
- 统一照片格式（jpg，短边>=400px）
- 建立猫猫名册：id/name/gender/neutered/personality/location
- 协助AI-1完成入库

**后端：完善数据库和接口**
- 数据库建表，插入初始猫猫数据
- 图片上传存储逻辑完成
- `/cats`和`/cats/{id}`接口对接真实数据库

**前端-1：对接mock接口**
- 上传 → loading → 展示结果的完整交互做通
- 三种状态UI：confirmed / uncertain / unknown

**前端-2：地图组件调研**
- 确定用高德地图还是Leaflet
- 地图组件跑起来，能显示一个点

---

### Day 3 — 第一次联调

**核心目标**：上传一张猫猫照片，页面上能看到识别结果。

**上午：AI接口上线**

AI-1将mock接口替换为真实识别接口：

```python
@app.post("/identify")
async def identify_cat(
    file: UploadFile,
    location_name: str = "",
    latitude: float = 0,
    longitude: float = 0
):
    image = load_image(await file.read())
    crop = crop_cat(image)
    features = extract_features(crop)
    scores, indices = search_cat(features, top_k=3)

    top_score = float(scores[0])
    status = classify_result(top_score)

    result = {
        "status": status,
        "confidence": top_score,
        "cat_id": None,
        "cat_name": None,
        "candidates": []
    }

    if status == "confirmed":
        cat = get_cat_by_index(indices[0])
        result["cat_id"] = cat.id
        result["cat_name"] = cat.name
        # 自动记录偶遇
        create_sighting(cat.id, location_name, latitude, longitude)

    elif status == "uncertain":
        result["candidates"] = [
            {"cat_id": get_cat_by_index(i).id,
             "cat_name": get_cat_by_index(i).name,
             "confidence": float(s)}
            for s, i in zip(scores[:3], indices[:3])
        ]

    return result
```

**下午：前后端联调**

- 前端切换到真实接口
- 测试三种状态是否正常展示
- 档案页对接真实猫猫数据

**今天结束验收**：能识别3只猫，档案页数据真实展示。

---

### Day 4 — 功能补全

**AI-2：全量入库**
- 所有猫猫照片处理完毕（目标覆盖猫协全部猫）
- 每只猫多角度照片取平均特征向量：
  ```python
  # 提升准确率的关键步骤
  all_features = [extract_features(photo) for photo in cat_photos]
  avg_feature = np.mean(all_features, axis=0)
  avg_feature = avg_feature / np.linalg.norm(avg_feature)
  faiss_index.add(avg_feature.reshape(1, -1))
  ```
- 测试边界case：模糊照片、侧脸、遮挡、两只颜色相近的猫

**后端：LLM故事生成**

```python
import anthropic

client = anthropic.Anthropic()

def generate_cat_story(cat):
    prompt = f"""
根据以下信息，用温暖有趣的语气写一段80-120字的猫猫介绍。
要求：像介绍一个有性格的老朋友，不要太正式，可以有一点点俏皮。

名字：{cat.name}
性别：{cat.gender}
是否绝育：{"是" if cat.neutered else "否"}
性格标签：{cat.personality}
常出没地点：{cat.location}
志愿者备注：{cat.notes}
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

**前端-1：完善交互细节**
- 上传时加进度动画（识别需要2-3秒，不能干等）
- uncertain状态下的候选卡片点击选择逻辑
- 档案页照片墙完成

**前端-2：偶遇动态流**
- 调用`/sightings`接口
- 时间线展示：谁在哪里看到了哪只猫
- 地图上能显示今日偶遇点位

---

### Day 5 — 集成打磨

**全员联调，重点测试主流程**：

```
✓ 上传照片 → 识别成功 → 自动记录偶遇 → 展示档案
✓ 上传照片 → 置信度中等 → 展示Top3候选 → 用户确认
✓ 上传照片 → 识别失败 → "发现新朋友！" 页面
✓ 首页 → 猫猫列表 → 点击进入档案页
✓ 动态流 → 今日谁在哪里被看见
✓ 地图 → 今日猫猫分布热力图
```

**AI-2：准备演示数据**
- 确认10张演示照片全部能正确识别
- 记录每只猫的识别准确率数字（答辩时用）

**后端：数据填充**
- 所有猫猫有完整档案和AI生成故事
- 造10-20条偶遇记录（覆盖不同地点和时间）

**前端：视觉打磨**
- 整体配色统一
- 移动端适配（答辩可能用手机演示）
- loading状态全部处理好，没有白屏

---

### Day 6 — 答辩准备

**上午：全流程预演**

按照以下脚本预演两遍，掐表控制在5分钟以内：

```
[0:00-0:30] 打开首页
  → 展示猫猫列表和地图热力图
  → 说一句："这是复旦校园里每一只猫的数字家园"

[0:30-1:30] 核心演示：识别
  → 上传一张刚拍的橘猫照片
  → 系统3秒内识别出"橘总"
  → 展示橘总档案：照片墙/性格故事/今日偶遇记录

[1:30-2:00] 展示动态流
  → 今天哪些同学在哪里偶遇了哪只猫

[2:00-2:30] 展示地图
  → 今日校园猫猫分布热力图

[2:30-3:30] 技术亮点
  → DINOv3多模态特征识别
  → 置信度分级的降级设计（展示uncertain和unknown状态）
  → 双层用户体系：猫协管理后台 vs 同学偶遇社区

[3:30-4:00] 社会价值
  → "猫协用它管理，同学用它偶遇，猫猫用它被记住"
  → 可扩展到其他高校校园猫保护场景
```

**下午：修最后的bug**
- 重点保证演示链路不能翻车
- 准备备用方案：如果现场网络差，准备离线版或录屏备份

---

### Day 7 — 上场

- 上午最后一次预演，确认所有人知道自己负责哪部分
- 下午答辩
- 演示时用提前准备好的照片，不要现场随机拍

---

## 五、风险与降级方案

### 最高风险：识别准确率不够

| 置信度 | 展示策略 |
|--------|---------|
| > 0.80 | 直接展示识别结果和档案 |
| 0.50–0.80 | 展示Top3候选，用户点击确认 |
| < 0.50 | "发现新朋友！帮它起个名字吧~" |

即使识别失败，用户体验依然流畅，答辩不会尬住。

### 次高风险：照片数据不足

如果猫协照片质量差或数量少：
- 数据增强：随机裁剪、色调抖动、模拟不同光线
- 降低演示预期：从"识别全校所有猫"改为"识别10只代表性猫猫"
- 答辩时强调"系统已具备扩展能力"

### 接口联调风险

- 后端第一天必须提供mock接口
- 前端不等后端，不等AI，全部对接mock先跑起来
- 出现接口不兼容，当天联调当天修

---

## 六、每日同步机制

每天晚上10分钟站会，每人说三件事：

1. 今天完成了什么
2. 明天计划做什么
3. 有没有被卡住的地方

**有人卡住立刻在群里说，不要憋到第二天。**

---

## 七、技术栈清单

| 模块 | 技术选型 |
|------|---------|
| 猫猫检测 | YOLOv8（ultralytics） |
| 特征提取 | DINOv3（timm: vit_base_patch16_dinov3 等） |
| 向量检索 | FAISS（faiss-cpu） |
| 后端框架 | FastAPI + SQLite（或PostgreSQL） |
| 图片存储 | 本地文件系统（OSS可选） |
| 故事生成 | Claude API（claude-sonnet-4-20250514） |
| 前端框架 | React + Tailwind CSS |
| 地图组件 | 高德地图JS SDK 或 Leaflet |
| 部署 | 本地演示 或 简单云端部署 |

---

## 八、附加分方案（可选）

如果有余力，可在Day 5-6实现：

将DINOv3识别模型部署到**本地端**（英特尔酷睿Ultra），构建混合架构：

```
用户上传照片
      ↓
本地端：YOLO裁图 + DINOv3特征提取（隐私保护，速度快）
      ↓
云端：FAISS检索 + 档案查询 + LLM故事生成
      ↓
返回结果
```

本地处理照片不上传原图，隐私保护是亮点，同时满足"混合架构"要求，拿附加分。

---

*最后一句话：今天联系猫协，没有真实猫猫数据，其他一切都是空的。*
