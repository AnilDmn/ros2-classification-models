import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ament_index_python.packages import get_package_share_directory


class CancerClassifierNode(Node):
    def __init__(self):
        super().__init__('cancer_classifier_node')
        self.publisher_ = self.create_publisher(String, 'cancer_classification_result', 10)
        
        # Set output directory path
        self.output_dir = "/home/anild/ros2_ml_ws/src/ml_classifiers_pkg"
        self.plots_dir = os.path.join(self.output_dir, 'plots', 'cancer')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.get_logger().info("Cancer Classifier Node started.")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.classify()

    def classify(self):
        """Main classification pipeline for breast cancer detection"""
        try:
            # Load breast cancer dataset
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)

            df = X.copy()
            df["label"] = y.map({0: "malignant", 1: "benign"})
            class_labels = ["malignant", "benign"]

            self.get_logger().info("Cancer dataset loaded successfully.")
            self.get_logger().info(f"Dataset shape: {X.shape}")
            self.get_logger().info(f"Number of features: {len(data.feature_names)}")
            
            # Initialize results dictionary for saving
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset_info': {
                    'shape': X.shape,
                    'features': len(data.feature_names),
                    'classes': class_labels,
                    'class_distribution': df['label'].value_counts().to_dict()
                }
            }

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.get_logger().info("Features standardized using StandardScaler")

            # Split dataset into training and testing sets (70/30 split)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y)
            
            self.get_logger().info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            results['data_split'] = {'train_size': X_train.shape[0], 'test_size': X_test.shape[0]}

            # Train SVM Linear Classifier
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.get_logger().info("SVM Linear Classifier trained successfully")

            # Evaluate model performance
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_labels)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Store results
            results['model_performance'] = {
                'accuracy': acc,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist()
            }

            # Log results
            self.get_logger().info("=== MODEL EVALUATION RESULTS ===")
            self.get_logger().info(f"SVM Accuracy: {acc:.4f}")
            self.get_logger().info(f"SVM Classification Report:\n{report}")

            # Publish result to ROS topic
            msg = String()
            msg.data = f"SVM Cancer Classification - Accuracy: {acc:.4f}"
            self.publisher_.publish(msg)

            # Generate and save visualizations
            self.plot_conf_matrix(conf_matrix, class_labels)
            self.plot_feature_distributions(df)
            self.plot_class_distribution(df)
            
            # Save results to text file
            self.save_results_to_txt(results, report, data.feature_names)

        except Exception as e:
            self.get_logger().error(f"Error during cancer classification: {e}")
            import traceback
            self.get_logger().error(f"Error details: {traceback.format_exc()}")

    def plot_conf_matrix(self, conf_matrix, class_labels):
        """Generate and save confusion matrix heatmap"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='d',
                        xticklabels=class_labels, yticklabels=class_labels,
                        cbar_kws={'label': 'Count'})
            plt.title("SVM Linear Classifier - Confusion Matrix\nBreast Cancer Classification")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()

            conf_path = os.path.join(self.plots_dir, "svm_confusion_matrix.png")
            plt.savefig(conf_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Confusion matrix saved to: {conf_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save confusion matrix: {e}")

    def plot_feature_distributions(self, df):
        """Generate and save feature distribution plots for selected features"""
        try:
            # Get feature columns (exclude label column)
            all_columns = list(df.columns)
            feature_columns = [col for col in all_columns if col != 'label']
            
            # Select first 8 most important features for visualization
            selected_features = feature_columns[:8]
            
            self.get_logger().info(f"Creating distribution plots for {len(selected_features)} features")
            
            for i, feature in enumerate(selected_features):
                plt.figure(figsize=(10, 6))
                
                # Create histogram for each class
                malignant_data = df[df['label'] == 'malignant'][feature].values
                benign_data = df[df['label'] == 'benign'][feature].values
                
                plt.hist(malignant_data, alpha=0.7, label='Malignant', bins=30, 
                        color='red', edgecolor='black', linewidth=0.5)
                plt.hist(benign_data, alpha=0.7, label='Benign', bins=30, 
                        color='blue', edgecolor='black', linewidth=0.5)
                
                plt.title(f"{feature} Distribution by Cancer Type")
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Create safe filename
                safe_filename = feature.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                dist_path = os.path.join(self.plots_dir, f"{safe_filename}_distribution.png")
                plt.savefig(dist_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.get_logger().info(f"{feature} distribution plot saved to: {dist_path}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to save feature distributions: {e}")
            import traceback
            self.get_logger().error(f"Error details: {traceback.format_exc()}")

    def plot_class_distribution(self, df):
        """Generate and save class distribution pie chart"""
        try:
            class_counts = df['label'].value_counts()
            
            plt.figure(figsize=(8, 8))
            colors = ['#ff9999', '#66b3ff']
            wedges, texts, autotexts = plt.pie(class_counts.values, 
                                              labels=class_counts.index,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90,
                                              explode=(0.05, 0.05))
            
            plt.title("Breast Cancer Dataset - Class Distribution")
            
            # Make percentage text bold and larger
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            plt.tight_layout()
            
            pie_path = os.path.join(self.plots_dir, "class_distribution.png")
            plt.savefig(pie_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Class distribution plot saved to: {pie_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save class distribution plot: {e}")

    def save_results_to_txt(self, results, classification_report, feature_names):
        """Save all results to a comprehensive text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'cancer_classification_results_{timestamp}.txt')
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("BREAST CANCER CLASSIFICATION RESULTS - SVM LINEAR CLASSIFIER\n")
                f.write("=" * 70 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dataset Shape: {results['dataset_info']['shape']}\n")
                f.write(f"Number of Features: {results['dataset_info']['features']}\n")
                f.write(f"Classes: {results['dataset_info']['classes']}\n")
                f.write(f"Class Distribution: {results['dataset_info']['class_distribution']}\n\n")
                
                f.write("FEATURE NAMES:\n")
                f.write("-" * 40 + "\n")
                for i, feature in enumerate(feature_names, 1):
                    f.write(f"{i:2d}. {feature}\n")
                f.write("\n")
                
                f.write("DATA PREPROCESSING:\n")
                f.write("-" * 40 + "\n")
                f.write("- Applied StandardScaler for feature normalization\n")
                f.write("- Train-test split: 70% training, 30% testing\n")
                f.write("- Used stratified sampling to maintain class balance\n\n")
                
                f.write("DATA SPLIT INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Training Size: {results['data_split']['train_size']} samples\n")
                f.write(f"Test Size: {results['data_split']['test_size']} samples\n\n")
                
                f.write("MODEL CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write("Algorithm: Support Vector Machine (SVM)\n")
                f.write("Kernel: Linear\n")
                f.write("Random State: 42\n\n")
                
                f.write("MODEL PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {results['model_performance']['accuracy']:.4f}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                conf_matrix = np.array(results['model_performance']['confusion_matrix'])
                f.write("           Predicted\n")
                f.write("         Malignant  Benign\n")
                f.write(f"Malignant    {conf_matrix[0][0]:3d}      {conf_matrix[0][1]:3d}\n")
                f.write(f"Benign       {conf_matrix[1][0]:3d}      {conf_matrix[1][1]:3d}\n\n")
                
                f.write("DETAILED CLASSIFICATION REPORT:\n")
                f.write("-" * 40 + "\n")
                f.write(classification_report)
                f.write("\n\n")
                
                f.write("GENERATED VISUALIZATIONS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"- Plots directory: {self.plots_dir}\n")
                f.write("- Generated files:\n")
                f.write("  * svm_confusion_matrix.png - Model performance visualization\n")
                f.write("  * class_distribution.png - Dataset class balance\n")
                f.write("  * Feature distribution plots for top 8 features:\n")
                
                # List feature distribution files
                feature_files = [f for f in os.listdir(self.plots_dir) if f.endswith('_distribution.png')]
                for file in sorted(feature_files):
                    if file != 'class_distribution.png':
                        f.write(f"    - {file}\n")
                
                f.write(f"\n- Results file: {results_file}\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 70 + "\n")
                
            self.get_logger().info(f"Comprehensive results saved to: {results_file}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save results to text file: {e}")


def main(args=None):
    """Main function to initialize and run the ROS2 node"""
    rclpy.init(args=args)
    node = CancerClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
