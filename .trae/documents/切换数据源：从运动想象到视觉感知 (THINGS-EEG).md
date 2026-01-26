# 双轨并行：视觉感知预训练 + 梦境数据微调

我完全理解你的担忧。你是对的，我之前的计划中对 Grok 提到的具体数据集（如 Dream2Image）重视不足。现在的计划将**显式地集成**这些关键资源，特别是 Grok 强烈推荐的 **Dream2Image** 和 **DREAM Database**，以及你提到的 **国内数据源**。

## 1. 核心策略：清醒学视觉，梦境学语义
我们将采用“双阶段”策略，这与 Grok 和 Qwen 的建议高度一致。
*   **Stage A (Pre-training)**: 使用 **THINGS-EEG** (清醒视觉)，让模型学会“看见物体时的脑电特征”。
*   **Stage B (Fine-tuning)**: 使用 **Dream2Image / DREAM** (真实梦境)，让模型学会“做梦时的脑电语义”。

## 2. 数据层重构 (Data Layer)
新建 `src/data_loader.py`，实现三个 Dataset 类：

1.  **`ThingsEEGDataset` (Stage A)**:
    *   **目标**: 学习通用的视觉-脑电映射。
    *   **来源**: THINGS-EEG (视觉诱发)。
    *   **实现**: 尝试自动下载 Sample，或生成符合视觉特征的 Dummy Data (重点关注枕叶通道)。

2.  **`Dream2ImageDataset` (Stage B - 首选)**:
    *   **目标**: 直接对齐梦境内容和 AI 生成图像。
    *   **来源**: Hugging Face `opsecsystems/Dream2Image` (如果可用) 或 arXiv 链接。
    *   **实现**: 尝试通过 `datasets` 库加载，如果失败则提供手动下载指引。

3.  **`DreamDatabaseDataset` (Stage B - 补充)**:
    *   **目标**: 利用大规模文本报告进行 CLIP 语义对齐。
    *   **来源**: **清华镜像** (`https://mirrors.tuna.tsinghua.edu.cn/dream-dataset/DREAM_mini.zip`)。
    *   **实现**: **直接集成 wget 下载代码**，解析 REM 片段和文本报告。

## 3. 模型与训练层更新
修改 `src/pipeline.py`：
*   **支持多模态目标**: 训练目标不再只是 Image Embedding，而是根据 Dataset 自动切换：
    *   THINGS -> Image Embedding
    *   Dream2Image -> Image Embedding
    *   DREAM -> Text Embedding (通过 CLIP Text Encoder 转为向量)
*   **迁移学习接口**: 增加 `load_pretrained_encoder()` 方法，允许加载 Stage A 的权重继续在 Stage B 训练。

## 4. 执行步骤
1.  **创建 `src/data_loader.py`**: 实现上述三个 Dataset 类，重点打通 **清华镜像下载**。
2.  **更新 `src/pipeline.py`**: 适配多数据源，实现“预训练-微调”逻辑。
3.  **更新 `requirements.txt`**: 加入 `datasets` (Hugging Face), `mne`, `braindecode` 等 Grok 提到的库。
4.  **更新 Colab**: 展示如何一键切换数据源（从清醒到梦境）。

这个计划**完全采纳**了 Grok 关于“迁移学习”和“特定数据集”的建议，并直接回应了你关于“清华镜像”和“梦境数据”的需求。我们将不再使用 MOABB 的运动数据。