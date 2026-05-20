# Visage Architecture

> macOS-native face clustering and photo sorting — from raw photos to organized folders, with an interactive review UI.

```mermaid
graph TB
    subgraph Input["输入层 Input Layer"]
        A[照片文件夹<br/>Photo Folder]
    end

    subgraph Pipeline["核心流水线 Core Pipeline (5阶段)"]
        direction TB
        P1["① 扫描 Scan<br/>遍历目录 查找图片"]
        P2["② 检测 Detect<br/>macOS Vision 人脸检测"]
        P3["③ 嵌入 Embed<br/>生成身份特征向量"]
        P4["④ 聚类 Cluster<br/>DBSCAN / HDBSCAN"]
        P5["⑤ 整理 Organize<br/>按人分文件夹"]
    end

    subgraph Storage["存储层 Storage"]
        C["SQLite 缓存<br/>Cache (嵌入向量)"]
        O["输出目录<br/>visage_output/"]
    end

    subgraph UI["交互界面 Review UI"]
        S["FastAPI 服务<br/>端口 8787"]
        F["React SPA<br/>前端页面"]
    end

    A --> P1
    P1 --> P2
    P2 --> P3
    P3 <--> C
    P3 --> P4
    P4 --> P5
    P5 --> O

    P3 -.->|--serve 模式| S
    P4 -.-> S
    S --> F
```

## 核心概念 Core Concepts

### 为何选择 macOS Vision 框架？

Visage 使用 macOS 原生的 Vision 框架进行人脸检测，而非 OpenCV 或深度学习模型：

- **硬件加速** — Vision 框架调用 Apple Neural Engine，速度优于纯软件方案
- **零模型依赖** — 无需下载或管理额外模型文件
- **高精度** — Apple 的检测器在大姿态、遮挡和光照变化下表现稳定

检测到的面部信息包括：
- **边界框** — 像素坐标 (top, right, bottom, left)
- **5点特征点** — 双眼、鼻尖、嘴角 (用于对齐)
- **头部轮廓** — 用于修正边界框，避免截断头顶/下巴

### 嵌入向量的作用

```mermaid
flowchart LR
    F1[人脸图像<br/>Face Image] --> A[对齐 Align<br/>仿射变换 112x112]
    A --> E[嵌入模型<br/>Embedding Model]
    E --> V1[向量 v128<br/>dlib 128维]
    E --> V2[向量 v512<br/>InsightFace 512维]
    V1 --> C[聚类 Clustering<br/>向量空间距离]
    V2 --> C
```

嵌入向量将人脸转换为数学表示：同一人的不同照片在向量空间中距离近，不同人的距离远。这是聚类的基础。

### 聚类的两个阶段

```
阶段一：初始聚类 (DBSCAN/HDBSCAN)
  ┌─────┐  ┌─────┐  ┌─────┐
  │  A  │  │  A  │  │  B  │  ← 可能过度分割
  │ 集群1│  │ 集群2│  │ 集群3│      (同人不同组)
  └─────┘  └─────┘  └─────┘
      │        │
      └────────┘
          │
          ▼
阶段二：后聚类合并 (余弦相似度)
  ┌─────────┐  ┌─────┐
  │    A    │  │  B  │  ← 合并后
  │  集群1  │  │ 集群3│
  └─────────┘  └─────┘
```

## 数据流 Data Flow

### 5阶段流水线详解

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI (cli.py)
    participant Scan as 扫描器 (scanner.py)
    participant Detect as 检测器 (detector.py)
    participant Embed as 嵌入器 (embedder.py)
    participant Cluster as 聚类器 (cluster.py)
    participant Org as 整理器 (organizer.py)
    participant Cache as SQLite缓存

    User->>CLI: visage ./照片/
    CLI->>Scan: scan_images()
    Scan-->>CLI: 图片路径列表 [path1, path2, ...]

    CLI->>Detect: detect_faces_batch()
    Detect-->>CLI: ImageResult[] (含 DetectedFace[])

    CLI->>Embed: generate_embeddings_batch()
    Embed->>Cache: 查询缓存
    Cache-->>Embed: 缓存命中? 跳过
    Embed-->>CLI: 含嵌入向量的 ImageResult[]

    CLI->>Cluster: extract_embeddings() → cluster_faces()
    Cluster-->>CLI: ClusterResult (labels, centroids)

    alt merge_threshold > 0
        CLI->>Cluster: merge_clusters() 后合并
    end

    CLI->>Org: build_organize_plan()
    Org-->>CLI: OrganizePlan
    CLI->>Org: execute_organize_plan()
    Org-->>User: person_00/ person_01/ ...
```

### Web UI 模式下的交互

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI
    participant WS as Workspace
    participant R as Routes

    Browser->>FastAPI: GET /api/pipeline-status (SSE)
    FastAPI-->>Browser: 阶段进度事件流

    Browser->>FastAPI: GET /api/workspace
    FastAPI->>WS: to_api_dict()
    WS-->>FastAPI: 包含所有聚类和照片的 JSON
    FastAPI-->>Browser: WorkspaceState

    Browser->>FastAPI: POST /api/clusters/merge {from_id, to_id}
    FastAPI->>WS: merge_clusters()
    WS-->>FastAPI: 更新后的状态 (含撤销栈)
    FastAPI-->>Browser: {ok: true, workspace: {...}}

    Browser->>FastAPI: POST /api/recluster {params...}
    FastAPI->>WS: get_recluster_data()
    FastAPI->>Cluster: cluster_faces() 重新聚类
    FastAPI-->>Browser: 新的 WorkspaceState

    Browser->>FastAPI: POST /api/save {output_dir, ...}
    FastAPI->>WS: save_to_disk()
    WS-->>Browser: {ok: true, stats: {...}}
```

## 组件关系 Component Relationships

```mermaid
classDiagram
    class VisageConfig {
        detection_confidence: float
        embedding_backend: str
        cluster_method: str
        merge_threshold: float
    }

    class PipelineResult {
        total_images: int
        num_clusters: int
        organize_plan: OrganizePlan
    }

    class ImageResult {
        path: str
        faces: DetectedFace[]
        error: str?
    }

    class DetectedFace {
        face_box: FaceBox
        embedding: ndarray?
        head_features: ndarray?
        landmarks_5: list?
    }

    class FaceBox {
        top: int
        right: int
        bottom: int
        left: int
    }

    class ClusterResult {
        labels: ndarray
        num_clusters: int
        num_noise: int
    }

    class Workspace {
        _cluster_mapping: dict
        _face_clusters: dict
        _history: list~Operation~
        merge_clusters()
        remove_face()
        move_face()
        save_to_disk()
        undo()
    }

    ImageResult "*" --> "*" DetectedFace
    DetectedFace --> "1" FaceBox
    PipelineResult --> "*" ImageResult
    PipelineResult --> ClusterResult
    PipelineResult --> OrganizePlan
    Workspace --> ClusterResult
    Workspace --> "*" ImageResult
```

## 关键设计决策 Key Design Decisions

### 1. 内存优先的内存模型

Workspace 类完全在内存中运行，不使用外部数据库：
- **优势**：零延迟操作、简单撤销实现（快照模式）
- **劣势**：数据集超大时可能受限
- **缓解**：sample_limit 配置项控制最大人脸数

### 2. 撤销栈 = O(1) 快照

每次修改操作前，Workspace 保存受影响数据的快照（而非完整状态）。撤销时直接恢复快照：

```
merge_clusters(from=1, to=2):
  1. 保存: {from_id: 1, to_id: 2, from_photos: [...], face_snapshot: {...}}
  2. 合并: _cluster_mapping[2] += _cluster_mapping[1]
  3. 删除: del _cluster_mapping[1]

undo():
  1. 弹出: {from_id: 1, ...}
  2. 恢复: _cluster_mapping[1] = from_photos
  3. 还原: _cluster_mapping[2] = to_photos_before
```

### 3. 人脸级别的聚类追踪

传统聚类按图片追踪（一张图 = 一个标签），Visage 支持"多人一张图"场景：

```
图片 photo.jpg 包含两张脸：
  face_index 0 → cluster 3 (Alice)
  face_index 1 → cluster 7 (Bob)

前端 ClusterDetail 仅显示属于该聚类的 face box：
  Alice 的详情页: 只画 face_index=0 的框
  Bob 的详情页: 只画 face_index=1 的框
  全部照片页: 两个框都画，各标各自的 cluster_id
```

### 4. 可插拔嵌入后端

```mermaid
flowchart LR
    subgraph Backends["嵌入后端"]
        D["dlib (默认)<br/>128维 / face_recognition"]
        I["InsightFace (可选)<br/>512维 / ArcFace"]
    end

    subgraph Cache["SQLite 缓存"]
        C1["key: (path + mtime)<br/>→ embedding"]
    end

    subgraph Config["自动配置"]
        H["hwdetect.py<br/>根据 RAM/CPU 推荐后端"]
    end

    H -->|低内存| D
    H -->|高内存| I
    D --> C
    I --> C
```

## 配置优先级 Configuration Priority

```
最高优先级
    ↑
 CLI 参数 (--eps 0.6 --backend insightface)
    ↑
 显式配置文件 (--config path/to/config.toml)
    ↑
 输入目录 visage.toml (自动发现)
    ↑
 硬件自适应推荐 (hwdetect.py)
    ↑
   最低优先级
 代码默认值 (VisageConfig dataclass)
```
