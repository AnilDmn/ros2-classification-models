import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ament_index_python.packages import get_package_share_directory


class IrisClassifierNode(Node):
    def __init__(self):
        super().__init__('iris_classifier_node')
        self.publisher_ = self.create_publisher(String, 'iris_classification_result', 10)
        
        # Specified output directory path
        self.output_dir = "/home/anild/ros2_ml_ws/src/ml_classifiers_pkg"
        self.plots_dir = os.path.join(self.output_dir, 'plots', 'iris')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.get_logger().info("Iris Classifier Node started.")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.classify_iris()

    def classify_iris(self):
        """Main classification pipeline for Iris dataset"""
        try:
            # Load the iris dataset
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = pd.Series(iris.target)
            df = X.copy()
            df['label'] = y

            # Map numeric labels to class names
            class_labels = iris.target_names
            label_map = dict(enumerate(class_labels))
            df['label'] = df['label'].map(label_map)

            self.get_logger().info("Iris dataset loaded successfully.")
            self.get_logger().info(f"Dataset shape: {X.shape}")
            self.get_logger().info(f"Feature columns: {list(X.columns)}")
            self.get_logger().info(f"Classes: {list(class_labels)}")

            # Initialize results dictionary for saving
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dataset_info': {
                    'shape': X.shape,
                    'features': list(X.columns),
                    'classes': list(class_labels),
                    'samples_per_class': df['label'].value_counts().to_dict()
                },
                'models': {}
            }

            # Normalize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.get_logger().info("Features normalized using StandardScaler")

            # Split the data into training and testing sets (70/30 split)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y)
            
            self.get_logger().info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            results['data_split'] = {'train_size': X_train.shape[0], 'test_size': X_test.shape[0]}

            # Define the machine learning models
            models = {
                'KNN': KNeighborsClassifier(n_neighbors=3),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
            }

            self.get_logger().info("=== MODEL TRAINING AND EVALUATION ===")
            
            # Train and evaluate each model
            for model_name, model in models.items():
                self.get_logger().info(f"Training {model_name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=class_labels)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # Store results
                results['models'][model_name] = {
                    'accuracy': acc,
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist()
                }

                # Log results
                self.get_logger().info(f"{model_name} Accuracy: {acc:.4f}")
                self.get_logger().info(f"{model_name} Classification Report:\n{report}")

                # Publish the result to ROS2 topic
                msg = String()
                msg.data = f"{model_name} Accuracy: {acc:.4f}"
                self.publisher_.publish(msg)

                # Generate and save confusion matrix plot
                self.plot_conf_matrix(model_name, conf_matrix, class_labels)

            # Generate additional visualizations
            self.plot_pairplot(df)
            self.plot_feature_distributions(df)
            
            # Save all results to text file
            self.save_results_to_txt(results)

        except Exception as e:
            self.get_logger().error(f"Error during classification: {e}")
            import traceback
            self.get_logger().error(f"Error details: {traceback.format_exc()}")

    def plot_conf_matrix(self, model_name, conf_matrix, class_labels):
        """Generate and save confusion matrix heatmap"""
        sns.set(style="whitegrid")
        try:
            # Create confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                        xticklabels=class_labels, yticklabels=class_labels,
                        cbar_kws={'label': 'Count'})
            plt.title(f"{model_name} - Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()

            # Save the plot
            save_path = os.path.join(self.plots_dir, f"{model_name.lower()}_confusion_matrix.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"{model_name} confusion matrix saved to: {save_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save confusion matrix plot for {model_name}: {e}")

    def plot_pairplot(self, df):
        """Generate and save pairplot showing relationships between features"""
        try:
            # Create pairplot
            sns.set(style="ticks")
            plt.figure(figsize=(12, 10))
            pairplot_fig = sns.pairplot(df, hue="label", diag_kind="hist", palette="husl")
            pairplot_fig.fig.suptitle("Iris Dataset - Feature Relationships", y=1.02)
            
            # Save the plot
            pairplot_path = os.path.join(self.plots_dir, "iris_pairplot.png")
            pairplot_fig.savefig(pairplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Pairplot saved to: {pairplot_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save pairplot: {e}")

    def plot_feature_distributions(self, df):
        """Generate and save feature distribution plots for each class"""
        try:
            # Get feature columns safely
            all_columns = list(df.columns)
            feature_columns = [col for col in all_columns if col != 'label']
            
            # Create a subplot for all features
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            # Create distribution plot for each feature
            for i, feature in enumerate(feature_columns):
                # Create separate histogram for each class
                for class_name in df['label'].unique():
                    class_data = df[df['label'] == class_name][feature].values
                    axes[i].hist(class_data, alpha=0.7, label=class_name, bins=15)
                
                axes[i].set_title(f"{feature} Distribution by Class")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle("Iris Dataset - Feature Distributions", fontsize=16)
            plt.tight_layout()
            
            # Save combined plot
            combined_path = os.path.join(self.plots_dir, "feature_distributions_combined.png")
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Combined feature distributions saved to: {combined_path}")
            
            # Create individual plots for each feature
            for feature in feature_columns:
                plt.figure(figsize=(10, 6))
                
                # Create separate histogram for each class
                for class_name in df['label'].unique():
                    class_data = df[df['label'] == class_name][feature].values
                    plt.hist(class_data, alpha=0.7, label=class_name, bins=20)
                
                plt.title(f"{feature} Distribution by Class")
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Create safe filename
                safe_filename = feature.replace(' ', '_').replace('(', '').replace(')', '')
                save_path = os.path.join(self.plots_dir, f"{safe_filename}_distribution.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.get_logger().info(f"{feature} distribution plot saved to: {save_path}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to save feature distribution plots: {e}")
            import traceback
            self.get_logger().error(f"Error details: {traceback.format_exc()}")

    def save_results_to_txt(self, results):
        """Save all classification results to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'iris_classification_results_{timestamp}.txt')
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("IRIS DATASET CLASSIFICATION RESULTS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dataset Shape: {results['dataset_info']['shape']}\n")
                f.write(f"Features: {results['dataset_info']['features']}\n")
                f.write(f"Classes: {results['dataset_info']['classes']}\n")
                f.write("Samples per Class:\n")
                for class_name, count in results['dataset_info']['samples_per_class'].items():
                    f.write(f"  - {class_name}: {count}\n")
                f.write("\n")
                
                f.write("DATA SPLIT:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Training Size: {results['data_split']['train_size']}\n")
                f.write(f"Test Size: {results['data_split']['test_size']}\n")
                f.write(f"Split Ratio: 70% train / 30% test\n\n")
                
                f.write("MODEL PERFORMANCE COMPARISON:\n")
                f.write("-" * 40 + "\n")
                
                # Summary table
                f.write("Model Performance Summary:\n")
                f.write(f"{'Model':<15} {'Accuracy':<10}\n")
                f.write("-" * 25 + "\n")
                for model_name, model_results in results['models'].items():
                    f.write(f"{model_name:<15} {model_results['accuracy']:<10.4f}\n")
                f.write("\n")
                
                # Detailed results for each model
                for model_name, model_results in results['models'].items():
                    f.write(f"{model_name.upper()} DETAILED RESULTS:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Accuracy: {model_results['accuracy']:.4f}\n\n")
                    
                    f.write("Confusion Matrix:\n")
                    conf_matrix = np.array(model_results['confusion_matrix'])
                    f.write("           Predicted\n")
                    f.write("         Setosa  Versicolor  Virginica\n")
                    f.write(f"Setosa      {conf_matrix[0][0]:2d}        {conf_matrix[0][1]:2d}        {conf_matrix[0][2]:2d}\n")
                    f.write(f"Versicolor  {conf_matrix[1][0]:2d}        {conf_matrix[1][1]:2d}        {conf_matrix[1][2]:2d}\n")
                    f.write(f"Virginica   {conf_matrix[2][0]:2d}        {conf_matrix[2][1]:2d}        {conf_matrix[2][2]:2d}\n\n")
                    
                    f.write("Detailed Classification Report:\n")
                    f.write(model_results['classification_report'])
                    f.write("\n" + "="*50 + "\n\n")
                
                f.write("GENERATED FILES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Plots Directory: {self.plots_dir}\n")
                f.write(f"Results File: {results_file}\n\n")
                f.write("Generated Visualizations:\n")
                f.write("- Confusion matrices for each model:\n")
                for model_name in results['models'].keys():
                    f.write(f"  * {model_name.lower()}_confusion_matrix.png\n")
                f.write("- Dataset visualizations:\n")
                f.write("  * iris_pairplot.png\n")
                f.write("  * feature_distributions_combined.png\n")
                f.write("  * Individual feature distribution plots\n")
                
                f.write("\nANALYSIS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                best_model = max(results['models'].items(), key=lambda x: x[1]['accuracy'])
                f.write(f"Best Performing Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})\n")
                
                accuracies = [model_results['accuracy'] for model_results in results['models'].values()]
                f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
                f.write(f"Standard Deviation: {np.std(accuracies):.4f}\n")
                
            self.get_logger().info(f"Comprehensive results saved to: {results_file}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save results to text file: {e}")


def main(args=None):
    """Main function to initialize and run the ROS2 node"""
    rclpy.init(args=args)
    node = IrisClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
