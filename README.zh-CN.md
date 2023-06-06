<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，由 [Ultralytics](https://ultralytics.com) 开发，是一种尖端的、最先进（SOTA）的模型，它在之前 YOLO 版本的成功基础上进行了建设，并引入了新的特性和改进，以进一步提高性能和灵活性。YOLOv8 旨在快速、准确且易于使用，使其成为广泛的对象检测、图像分割和图像分类任务的绝佳选择。

如需申请企业许可，请在 [Ultralytics 授权](https://ultralytics.com/license) 完成表格。

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
</div>
</div>

## <div align="center">文档</div>

请参阅下面的快速安装和使用示例，以及 [YOLOv8 文档](https://docs.ultralytics.com) 上有关培训、验证、预测和部署的完整文档。

<details open>
<summary>安装</summary>

在一个 [**Python>=3.7**](https://www.python.org/) 环境中，使用 [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)，通过 pip 安装 ultralytics 软件包以及所有[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。

```bash
pip install ultralytics
```

</details>

<details open>
<summary>Usage</summary>

#### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

#### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco128.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)中下载。查看 YOLOv8 [Python 文档](https://docs.ultralytics.com/usage/python)以获取更多示例。

</details>

## <div align="center">模型</div>

所有的 YOLOv8 预训练模型都可以在此找到。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml) 数据集上进行预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/ImageNet.yaml) 数据集上进行预训练。

在首次使用时，[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)中下载。

<details open><summary>检测</summary>

查看 [检测文档](https://docs.ultralytics.com/tasks/detect/) 以获取使用这些模型的示例。

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val detect data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val detect data=coco128.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>分割</summary>

查看 [分割文档](https://docs.ultralytics.com/tasks/segment/) 以获取使用这些模型的示例。

| 模型                                                                                           | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640             | 36.7                 | 30.5                  | 96.1                        | 1.21                             | 3.4            | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640             | 44.6                 | 36.8                  | 155.7                       | 1.47                             | 11.8           | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640             | 49.9                 | 40.8                  | 317.0                       | 2.18                             | 27.3           | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640             | 52.3                 | 42.6                  | 572.4                       | 2.79                             | 46.0           | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640             | 53.4                 | 43.4                  | 712.1                       | 4.02                             | 71.8           | 344.1             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val segment data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>分类</summary>

查看 [分类文档](https://docs.ultralytics.com/tasks/classify/) 以获取使用这些模型的示例。

| 模型                                                                                           | 尺寸<br><sup>(像素) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------- | ---------------- | ---------------- | --------------------------- | -------------------------------- | -------------- | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224             | 66.6             | 87.0             | 12.9                        | 0.31                             | 2.7            | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224             | 72.3             | 91.1             | 23.4                        | 0.35                             | 6.4            | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224             | 76.4             | 93.2             | 85.4                        | 0.62                             | 17.0           | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224             | 78.0             | 94.1             | 163.0                       | 0.87                             | 37.5           | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224             | 78.4             | 94.3             | 232.0                       | 1.01                             | 57.4           | 154.8                    |

- **acc** 值是模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。
  <br>通过 `yolo val classify data=path/to/ImageNet device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 ImageNet val 图像进行平均计算的。
  <br>通过 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现

</details>

<details><summary>姿态</summary>

查看 [姿态文档](https://docs.ultralytics.com/tasks/) 以获取使用这些模型的示例。

| 模型                                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------- | --------------------- | ------------------ | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640             | 49.7                  | 79.7               | 131.8                       | 1.18                             | 3.3            | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640             | 59.2                  | 85.8               | 233.2                       | 1.42                             | 11.6           | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640             | 63.6                  | 88.8               | 456.3                       | 2.00                             | 26.4           | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640             | 67.0                  | 89.9               | 784.5                       | 2.59                             | 44.4           | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640             | 68.9                  | 90.4               | 1607.1                      | 3.73                             | 69.4           | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280            | 71.5                  | 91.3               | 4088.7                      | 10.04                            | 99.1           | 1066.4            |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO Keypoints val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val pose data=coco-pose.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu` 复现

</details>

## <div align="center">集成</div>

<br>
<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" /></a>
</div>

|                                      Roboflow                                      |                                 ClearML ⭐ NEW                                  |                                     Comet ⭐ NEW                                      |                                  Neural Magic ⭐ NEW                                   |
| :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
| 使用 [Roboflow](https://roboflow.com/?ref=ultralytics) 将您的自定义数据集直接标记并导出至 YOLOv8 进行训练 | 使用 [ClearML](https://cutt.ly/yolov5-readme-clearml)（开源！）自动跟踪、可视化，甚至远程训练 YOLOv8 | 免费且永久，[Comet](https://bit.ly/yolov8-readme-comet) 让您保存 YOLOv8 模型、恢复训练，并以交互式方式查看和调试预测 | 使用 [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) 使 YOLOv8 推理速度提高多达 6 倍 |

## <div align="center">Ultralytics HUB</div>

体验 [Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐ 带来的无缝 AI，这是一个一体化解决方案，用于数据可视化、YOLOv5 和即将推出的 YOLOv8 🚀 模型训练和部署，无需任何编码。通过我们先进的平台和用户友好的 [Ultralytics 应用程序](https://ultralytics.com/app_install)，轻松将图像转化为可操作的见解，并实现您的 AI 愿景。现在就开始您的**免费**之旅！

<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png"></a>

## <div align="center">贡献</div>

我们喜欢您的参与！没有社区的帮助，YOLOv5 和 YOLOv8 将无法实现。请参阅我们的[贡献指南](CONTRIBUTING.md)以开始使用，并填写我们的[调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)向我们提供您的使用体验反馈。感谢所有贡献者的支持！🙏

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png"></a>

## <div align="center">许可证</div>

YOLOv8 提供两种不同的许可证：

- **AGPL-3.0 许可证**：详细信息请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
- **企业许可证**：为商业产品开发提供更大的灵活性，无需遵循 AGPL-3.0 的开源要求。典型的用例是将 Ultralytics 软件和 AI 模型嵌入商业产品和应用中。在 [Ultralytics 授权](https://ultralytics.com/license) 处申请企业许可证。

## <div align="center">联系方式</div>

如需报告 YOLOv8 的错误或提出功能需求，请访问 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 或 [Ultralytics 社区论坛](https://community.ultralytics.com/)。

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="" /></a>
</div>
