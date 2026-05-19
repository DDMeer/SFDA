# Source-Free Domain Adaptation (SFDA) via Fiber Bundle Disentanglement and Active Learning

本项目基于纤维丛理论 (Fiber Bundle Theory)，构建了一个跨域的无源领域自适应 (SFDA) 生成与重构框架。系统能够从源域（如：艺术油画 `art_painting`）中解耦出不随风格改变的**几何拓扑底座 ($z_c$)**，并在目标域（如：写实照片 `photo`）中利用生成模型重塑其视觉纤维 ($z_s$)。最终，通过主动学习 (Active Learning)错题本机制实现模型的自我进化。

---

## 核心架构与 4 阶段逻辑 (Four-Stage Pipeline)

项目完全契合“底座与纤维”的数学构造，代码分为以下四个演进阶段：

### Stage 1: 几何底座解耦 (Base Space Disentanglement)
* 核心模型: 简化流模型 (`SimplifiedGlow`)
* 核心逻辑: 输入源域图像，利用可逆流模型的强大密度估计能力，将图像强行剥离为两部分：承载构图、拓扑边界的**几何底座 $z_c$ (Content Skeleton)**，以及承载笔触、色彩的环境**风格纤维 $z_s$ (Style Fiber)**。

### Stage 2: 灵魂概念映射 (Concept Bridge Alignment)
* 核心模型: 语义桥梁网络 (`ConceptBridge`) + CLIP (ViT-B/32)
* 核心逻辑: 将低分辨率的 $z_c$ 骨架进行通道自适应补齐 (`Padding`)，通过多层感知机 (MLP) 强行映射至 CLIP 的文本特征空间。计算与 7 个类别提示词的余弦相似度，从而在缺失目标域标签的情况下，为骨架赋予准确的**灵魂语义标签 (Semantic Label)**。

### Stage 3: 结构保形重塑 (Geometric Preserving Reconstruction)
* 核心模型: Stable Diffusion v1-5 (Img2Img Pipeline)
* 核心逻辑: 执行“跨域换肤手术”。保持 $z_c$ 不变，将风格 $z_s$ 置零逆推回图像空间，制造出清晰的“灰色影子轮廓图”作为几何引导模具；结合 Stage 2 识别出的语义 Prompt，驱动 SD 顺着轮廓编织目标域（写实照片）的全新皮肤质感，实现真正的保形跨域重构。

### Stage 4: 主动学习进化 (Active Learning Error-Book Mechanism)
* 核心逻辑: 构建自我进化的“错题本”。系统计算目标域图像在分类器中的预测熵 (Entropy)。高熵样本意味着模型处于“极度迷茫”状态（如置信度仅为随机盲猜的 0.15）。系统会自动挖掘这些“黄金困难样本 (Hard Samples)”，将其挑出并采用 Stage 3 管道重点重塑与针对性强化训练，大幅降低领域迁移的整体错误率。

---

## 环境配置



### 1. 本地生成环境说明书
conda create -n sfda_env python=3.10 -y
conda activate sfda_env
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git