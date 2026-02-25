# YOLO-CTEx
YOLO-CTEx: Anatomy-Guided Semi-Supervised Detection for Acute Exacerbation Characteristics in Chest CT with Limited Annotations
We present YOLO-CTEx, a lightweight semisupervised framework designed specifically for detecting COPD-related features in chest CT images. YOLO-CTEx optimizes the YOLOv10n architecture with a lung-masked guided learning approach, effectively reducing model size and computational complexity without large labeled data. The contributions of this
study are as follows:
1) We propose a lightweight YOLOv10n version with two key innovations, reducing model parameters by 57%, while preserving competitive performance in detecting emphysema and BWT tasks.
2)We introduce YOLO-CTEx, a semi-supervised framework capable of achieving outperforming performance using only 10% labeled data, significantly reducing the annotation burden for clinical deployment.
3) We construct an in-house dataset focused on emphysema and BWT, consisting of 6,872 labeled chest CT slices, cross-validated by two expert radiologists to ensure annotation consistency and reliability.

The proposed YOLO-CTEx framework is designed to efficiently detect two key features of AECOPD, emphysema and bronchial wall thickening (BWT), in chest CT scans. These features are critical for early diagnosis but challenging to annotate, especially by junior radiologists, leading to limited labeled data. To address this, YOLO-CTEx integrates three main components: (1) a lightweight YOLOv10n optimized with smaller convolution kernels and efficient channel attention mechanisms, resulting in a 57% reduction in parameters while maintaining fast, acceptable accurate, and scalable detection; (2) a semi-supervised learning strategy using both labeled and unlabeled data with lung-masked guided
augmentations to enhance model generalization and focus on lung regions and structures; and (3) pretraining on the large-scale DeepLesion dataset, which consists
of axial CT slices, to improve feature extraction capabilities.

