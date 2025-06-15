# ROS 2 CLASSSIFICATION MODELS
A ROS 2 Python package for training and running machine learning classifiers on various public datasets (Iris, Penguins, Fruits, and Breast Cancer). The package provides ROS 2 nodes for inference and visualization, along with accompanying plots and logs.

Features
Dataset support: Iris, Penguins, Breast Cancer, Fruits

ML models: SVM, KNN, Decision Tree, Random Forest

Visualization: Confusion matrices, feature distributions, pairplots

ROS 2 nodes: For classification tasks on each dataset

Logging: Model performance and inference results are logged

# Project Structure

ml_classifiers_pkg/
├── plots/                          # Saved visualizations
│   ├── confusion_matrix.png
│   ├── feature_distributions.png
│   └── weight_vs_sphericity.png
├── results/                        # Classification results
│   └── fruit_classification_results_*.txt
├── ml_classifiers_pkg/            # Python package (ROS 2 node implementations)
│   ├── __init__.py
│   ├── cancer_node.py
│   ├── fruit_node.py
│   ├── iris_node.py
│   └── penguin_node.py
├── data/                           # Input datasets
│   ├── Breast_Cancer.csv
│   ├── Iris.csv
│   ├── Penguin.csv
│   └── fruits_weight_sphercity.csv
├── package.xml                    # ROS 2 package manifest
├── setup.py                       # Python setup script for ROS 2
└── plots/                         # All generated visualizations organized by dataset

Requirements
ROS 2 Humble or newer
Python 3.10+

Dependencies:
numpy
pandas
scikit-learn
seaborn
matplotlib
rclpy

Install them with:
pip install -r requirements.txt

Build Instructions:
# Source ROS 2
source /opt/ros/humble/setup.bash

# Navigate to workspace
cd ~/ros2_ml_ws

# Build with colcon
colcon build --packages-select ml_classifiers_pkg

# Source the overlay
source install/setup.bash

Running Nodes:

# Cancer Classification
ros2 run ml_classifiers_pkg cancer_node

# Fruit Classification
ros2 run ml_classifiers_pkg fruit_node

# Iris Classification
ros2 run ml_classifiers_pkg iris_node

# Penguin Classification
ros2 run ml_classifiers_pkg penguin_node

Output:
Inference results are stored in results/
Plots are saved in plots/ and install/ml_classifiers_pkg/share/ml_classifiers_pkg/plots/
Typical outputs include:
Confusion matrices
Feature histograms
Correlation matrices
Scatter and pair plots

Notes:
Nodes read datasets from the data/ folder and save results in results/.
