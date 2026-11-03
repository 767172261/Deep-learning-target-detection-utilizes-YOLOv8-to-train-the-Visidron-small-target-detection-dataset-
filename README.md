深度学习目标检测 利用YOLOv8训练Visidron小目标检测数据集通过训练自己的pt权重模型。无人机车辆行人数据集的训练应用 识别无人机视角下的车辆行人检测。
@[toc]

深度学习目标检测 利用YOLOv8训练Visidron小目标检测数据集并做出来自己的pt权重模型![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f0c932b9fc07494692f6a2ef444e44fb.png)
在深度学习目标检测中如何使用YOLOv8训练Visidron小目标检测数据集和自己的权重并作出自己的训练参数

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78d49b23702841bdaaf05f035dcb4b37.png)

训练好的YOLO-visdrone.pt 两个模型

2、包含各种训练曲线
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b0b1fcf15f004609af1de6c666dc7d4f.png)
YOLO小目标检测精度提升，加入小目标的anchor参数，neck加入小尺寸层，变为4个detect，添加各自注意力机制，修改检测头等![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d29e92a76cec4ddc915c2479e6786a1f.png)


使用YOLOv8来训练Visidron小目标检测数据集。以下是详细的步骤：

1. **环境准备**：
   - 安装必要的库。
   - 下载并组织数据集。

2. **数据预处理**：
   - 确认数据集的划分和标签格式。
   - 格式化标签文件以便于YOLOv8使用（如果需要）。

3. **模型定义与训练**：
   - 使用YOLOv8进行目标检测。

4. **评估与可视化**：
   - 评估模型性能。
   - 可视化结果。

### 环境准备

首先，我们需要安装必要的库。您可以使用以下命令来设置环境：

```bash
pip install ultralytics
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/38dc1125dfa0460d84c9460f2e42392e.png)

### 数据预处理

假设您的数据集已经下载并存储在`datasets/visidron`目录中，其中包含`images`和`labels`子目录。`labels`子目录中的每个`.txt`文件对应一个图像文件，格式为YOLOv5/YOLOv8所需的格式。

以下是确认数据集有效性和检查标签格式的脚本 `check_dataset.py`:

```python
[<title="Check Dataset Script for Visidron Small Object Detection">]
import os
from pathlib import Path

# Define paths
base_path = Path('datasets/visidron')
train_images_dir = base_path / 'train/images'
train_labels_dir = base_path / 'train/labels'
val_images_dir = base_path / 'val/images'
val_labels_dir = base_path / 'val/labels'
test_images_dir = base_path / 'test/images'
test_labels_dir = base_path / 'test/labels'

# Function to check if label files exist for all images
def check_labels_exist(images_dir, labels_dir):
    missing_labels = []
    for img_file in images_dir.glob('*.jpg'):  # Adjust extension if necessary
        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            missing_labels.append(img_file.name)
    return missing_labels

# Check train set
missing_train_labels = check_labels_exist(train_images_dir, train_labels_dir)
if missing_train_labels:
    print(f"Missing train labels: {missing_train_labels}")
else:
    print("All train labels are present.")

# Check val set
missing_val_labels = check_labels_exist(val_images_dir, val_labels_dir)
if missing_val_labels:
    print(f"Missing val labels: {missing_val_labels}")
else:
    print("All val labels are present.")

# Check test set
missing_test_labels = check_labels_exist(test_images_dir, test_labels_dir)
if missing_test_labels:
    print(f"Missing test labels: {missing_test_labels}")
else:
    print("All test labels are present.")
```

### 创建YAML配置文件

确保您有一个正确的YAML配置文件 `visidron.yaml` 来描述数据集。以下是示例配置文件：

```yaml
[<title="YOLOv8 Configuration File for Visidron Small Object Detection">]
train: ../datasets/visidron/train/images
val: ../datasets/visidron/val/images

nc: 6
names: ['general cargo ship', 'fishing boat', 'ore carrier', 'container ship', 'bulk cargo carrier', 'passenger ship']
```

将上述内容保存为 `datasets/visidron/visidron.yaml`。

### 模型定义与训练

我们将使用YOLOv8进行目标检测。以下是训练脚本 `train_detection.py`:

```python
[<title="Training Script for Visidron Small Object Detection using YOLOv8">]
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='../datasets/visidron/visidron.yaml',
    epochs=50,
    imgsz=1080,
    batch=16,
    project='../runs/train',
    name='visidron_small_object_detection'
)

# Evaluate the model
metrics = model.val()
results = model.export(format='onnx')  # export the trained model to ONNX format
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d58804c3e6a4ce09d2835984cf95c7e.png)

### 评估与可视化

使用YOLOv8自带的评估脚本来评估目标检测模型。

```python
[<title="Evaluation Script for Visidron Small Object Detection using YOLOv8">]
from ultralytics import YOLO

# Load the best model
best_model = YOLO('../runs/train/visidron_small_object_detection/weights/best.pt')

# Evaluate the model on the validation dataset
metrics = best_model.val(data='../datasets/visidron/visidron.yaml', conf=0.5, iou=0.45)
print(metrics)
```

### 用户界面

我们将使用 PyQt5 创建一个简单的 GUI 来加载和运行模型进行实时预测。以下是用户界面脚本 `ui.py`:

```python
[<title="PyQt5 Main Window for Visidron Small Object Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

# Load model
detection_model = YOLO('../runs/train/visidron_small_object_detection/weights/best.pt')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visidron 小目标检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("加载图像", self)
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        self.start_prediction_button = QPushButton("开始预测", self)
        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.start_prediction_button)

        self.stop_prediction_button = QPushButton("停止预测", self)
        self.stop_prediction_button.clicked.connect(self.stop_prediction)
        self.layout.addWidget(self.stop_prediction_button)

        self.central_widget.setLayout(self.layout)

        self.image_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.display_image(file_name)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def start_prediction(self):
        if self.image_path is not None and not self.timer.isActive():
            self.timer.start(30)  # Update frame every 30 ms

    def stop_prediction(self):
        if self.timer.isActive():
            self.timer.stop()
            self.image_label.clear()

    def update_frame(self):
        original_image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Detection
        results = detection_model.predict(image_rgb, size=1080, conf=0.5, iou=0.45)[0]

        for box in results.boxes.cpu().numpy():
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Map class ID to name
            class_names = ['普通货船', '捕鱼船', '矿石运输船', '集装箱船', '散货船', '客船']
            class_name = class_names[cls]

            # Draw bounding box
            cv2.rectangle(image_rgb, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_rgb, f'{class_name} ({conf:.2f})', (r[0], r[1] - 10), font, 0.9, (0, 255, 0), 2)

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

请确保将路径替换为您实际的路径。

### 使用说明

1. **配置路径**：
   - 确保 `datasets/visidron` 目录结构正确，并且包含 `train`、`val` 和 `test` 子目录。
   - 确保 `runs/train/visidron_small_object_detection/weights/best.pt` 是训练好的 YOLOv8 模型权重路径。

2. **运行脚本**：
   - 在终端中运行 `check_dataset.py` 脚本来检查数据集的有效性。
   - 在终端中运行 `train_detection.py` 脚本来训练目标检测模型。
   - 在终端中运行 `evaluate_detection.py` 来评估目标检测模型性能。
   - 在终端中运行 `ui.py` 来启动 GUI 应用程序。

3. **注意事项**：
   - 确保所有必要的工具箱已安装，特别是 `ultralytics` 和 `PyQt5`。
   - 根据需要调整参数，如 `epochs` 和 `batch_size`。

### 示例

假设您的数据文件夹结构如下：
```
datasets/
└── visidron/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

并且每个数据集中包含相应的图像和标签文件。运行 `ui.py` 后，您可以点击按钮来加载图像并进行Visidron小目标检测。


