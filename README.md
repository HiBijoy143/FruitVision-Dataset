# FruitVision-Dataset

# Version: v1.0
- Last Updated: April 2025
- License: MIT License

# Overview
The FruitVision Dataset is a high-quality, expert-validated image dataset designed for the classification of fruits under different physical states: fresh, rotten, and formalin-mixed (toxic). It is intended for use in computer vision research, machine learning model development, and agricultural health informatics applications. The dataset includes images of Apple, Banana, Mango, Orange, and Grapes captured in real-world conditions using different smartphone devices and lighting setups. It also provides a wide variety of augmentations and preprocessing for better generalization in model training.

# Metadata

| Attribute              | Description |
|------------------------|-------------|
| **Dataset Name**       | FruitVision Dataset |
| **Version**            | v1.0 |
| **Last Updated**       | April 2025 |
| **Fruits Included**    | Apple, Banana, Mango, Orange, Grapes |
| **Categories**         | Fresh, Rotten, Formalin-Mixed (Toxic) |
| **Total Images**       | [Insert total count here] |
| **File Format**        | .jpg |
| **Resolution**         | 512 × 512 pixels |
| **Capture Devices**    | iPhone 15 Pro Max, Redmi POCO M2 Reloaded, Redmi Note 9 Pro |
| **Capture Conditions** | Daylight, Indoor lighting, Plain backgrounds |
| **Annotation**         | Manual, verified by agricultural domain experts |
| **Augmentation**       | Rotation, Flip, Zoom, Brightness, Shear, Gaussian Noise |
| **Preprocessing**      | Resizing to 512×512 using OpenCV |
| **Intended Use**       | Fruit classification, spoilage detection, food safety ML |


# Preprocessing and Augmentation
- Augmentation performed using imgaug and OpenCV libraries includes:

1. Rotation: 45°, 60°, 90°
2. Flip: Horizontal Flip (50% chance)
3. Zooming: Scale range 0.8–1.2
4. Brightness: Multiplication by 0.8–1.2
5. Shearing: Range from -16 to +16 degrees
6. Noise: Additive Gaussian Noise

# Download dataset: 
- Link: https://data.mendeley.com/datasets/xkbjx8959c/2

# Citation: 
Bijoy, Md Hasan Imam; Tasnim, Syeda Zarin; Awsaf, Syed Ali; Hasan, Md Zahid (2025), “FruitVision: A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection”, Mendeley Data, V2, doi: 10.17632/xkbjx8959c.2

# Ethical Statement: 
All procedures for the fruit data collection were conducted in compliance with Daffodil International University's (DIU) ethical guidelines and relevant regulations. Ethical approval was granted by the Research Ethics Committee, Faculty of Science and Information Technology, DIU, under the approval number REC-FSIT-2024-04-17. This approval followed a comprehensive review process, ensuring adherence to safety protocols and ethical standards. The formalin-mixed fruits were strictly used for research purposes, validated by agricultural experts, and were not intended for consumption or sale to consumers

# Acknowledgement
We extend our heartfelt gratitude to Professor Dr. M. A. Rahim, Head of the Department of Agricultural Science at Daffodil International University (DIU), Dhaka, Bangladesh, for his invaluable expertise in data validation. His insightful feedback and unwavering support were instrumental in the successful completion of this project.


