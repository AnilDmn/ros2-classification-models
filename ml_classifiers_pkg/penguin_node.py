import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ament_index_python.packages import get_package_share_directory


class PenguinClassifierNode(Node):
    def __init__(self):
        super().__init__('penguin_classifier_node')
        self.publisher_ = self.create_publisher(String, 'penguin_classification_result', 10)
        
        # Set output directory to the specified WSL path
        self.output_dir = "/home/anild/ros2_ml_ws/src/ml_classifiers_pkg"
        self.plots_dir = os.path.join(self.output_dir, 'plots', 'penguin')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.get_logger().info("Penguin Classifier Node started.")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.classify()

    def classify(self):
        """Main classification pipeline for penguin species prediction"""
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'SVM Linear Classifier',
            'dataset_info': {}
        }
        
        try:
            # Load penguin dataset from CSV file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(current_dir, '..', '..', '..', 'src', 'ml_classifiers_pkg', 'datasets', 'penguins.csv')
            
            # Alternative path if above doesn't work
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(os.path.expanduser('~'), 'ros2_ml_ws', 'src', 'ml_classifiers_pkg', 'datasets', 'penguins.csv')
            
            # If still not found, try loading from seaborn as fallback
            if not os.path.exists(dataset_path):
                self.get_logger().warning("CSV file not found, loading from seaborn as fallback")
                df = sns.load_dataset('penguins')
                results['dataset_source'] = 'seaborn built-in dataset'
            else:
                df = pd.read_csv(dataset_path)
                results['dataset_source'] = dataset_path
            
            if df is None or df.empty:
                self.get_logger().error("Failed to load penguin dataset")
                return
            
            self.get_logger().info(f"Penguin dataset loaded successfully. Shape: {df.shape}")
            self.get_logger().info(f"Dataset columns: {list(df.columns)}")
            self.get_logger().info(f"Missing values:\n{df.isnull().sum()}")
            
            # Store initial dataset info
            results['dataset_info']['original_shape'] = df.shape
            results['dataset_info']['columns'] = list(df.columns)
            results['dataset_info']['missing_values'] = df.isnull().sum().to_dict()
            
            # Clean missing values
            df = df.dropna()
            self.get_logger().info(f"After removing missing values. Shape: {df.shape}")
            results['dataset_info']['cleaned_shape'] = df.shape
            
            # Feature engineering
            # Encode categorical variables
            label_encoders = {}
            categorical_columns = ['island', 'sex']
            
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col])
                    label_encoders[col] = le
            
            # Define features and target
            feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                             'body_mass_g', 'island_encoded', 'sex_encoded']
            
            X = df[feature_columns]
            y = df['species']
            
            # Encode target variable
            species_encoder = LabelEncoder()
            y_encoded = species_encoder.fit_transform(y)
            
            self.get_logger().info(f"Species distribution:\n{df['species'].value_counts()}")
            results['dataset_info']['species_distribution'] = df['species'].value_counts().to_dict()
            results['dataset_info']['features_used'] = feature_columns
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
            
            results['data_split'] = {
                'train_size': X_train.shape[0], 
                'test_size': X_test.shape[0],
                'test_ratio': 0.3
            }
            
            # Train SVM Linear Classifier
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate model performance
            acc = accuracy_score(y_test, y_pred)
            class_labels = species_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_labels)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store performance metrics
            results['model_performance'] = {
                'accuracy': acc,
                'class_labels': class_labels.tolist(),
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist()
            }
            
            self.get_logger().info(f"SVM Accuracy: {acc:.4f}")
            self.get_logger().info(f"SVM Classification Report:\n{report}")
            
            # Publish result
            msg = String()
            msg.data = f"SVM Penguin Accuracy: {acc:.4f}"
            self.publisher_.publish(msg)
            
            # Generate visualizations
            self.plot_conf_matrix(conf_matrix, class_labels)
            self.plot_feature_distributions(df)
            self.plot_species_by_island(df)
            self.plot_correlation_matrix(df[feature_columns + ['species']])
            
            # Save results to text file
            self.save_results_to_txt(results, report)
            
        except Exception as e:
            self.get_logger().error(f"Error during penguin classification: {e}")
            import traceback
            self.get_logger().error(f"Error details: {traceback.format_exc()}")

    def plot_conf_matrix(self, conf_matrix, class_labels):
        """Generate and save confusion matrix heatmap"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.title("SVM - Penguin Species Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            path = os.path.join(self.plots_dir, "svm_confusion_matrix.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Confusion matrix saved to: {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save confusion matrix: {e}")

    def plot_feature_distributions(self, df):
        """Generate and save feature distribution plots for each species"""
        try:
            # Numerical features for distribution plots
            numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
            
            for feature in numerical_features:
                plt.figure(figsize=(10, 6))
                
                # Create histogram for each species
                for species in df['species'].unique():
                    species_data = df[df['species'] == species][feature].values
                    plt.hist(species_data, alpha=0.7, label=species, bins=20)
                
                plt.title(f"{feature} Distribution by Species")
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Create safe filename
                safe_filename = feature.replace('_', '').replace('mm', '').replace('g', '')
                path = os.path.join(self.plots_dir, f"{safe_filename}_distribution.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                self.get_logger().info(f"{feature} distribution plot saved to: {path}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to save feature distributions: {e}")

    def plot_species_by_island(self, df):
        """Generate and save species distribution by island plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create species count by island bar plot
            species_island = df.groupby(['island', 'species']).size().unstack(fill_value=0)
            species_island.plot(kind='bar', ax=plt.gca())
            
            plt.title("Penguin Species Distribution by Island")
            plt.xlabel("Island")
            plt.ylabel("Count")
            plt.legend(title="Species")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            path = os.path.join(self.plots_dir, "species_by_island.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Species by island plot saved to: {path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save species by island plot: {e}")

    def plot_correlation_matrix(self, df):
        """Generate and save feature correlation matrix heatmap"""
        try:
            # Calculate correlation only for numerical features
            numerical_df = df.select_dtypes(include=[np.number])
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = numerical_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f')
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()

            path = os.path.join(self.plots_dir, "correlation_matrix.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Correlation matrix saved to: {path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save correlation matrix: {e}")

    def save_results_to_txt(self, results, classification_report):
        """Save comprehensive results to text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'penguin_classification_results_{timestamp}.txt')
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("PENGUIN SPECIES CLASSIFICATION RESULTS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n")
                f.write(f"Model Type: {results['model_type']}\n")
                f.write(f"Dataset Source: {results.get('dataset_source', 'Unknown')}\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Original Shape: {results['dataset_info']['original_shape']}\n")
                f.write(f"Cleaned Shape: {results['dataset_info']['cleaned_shape']}\n")
                f.write(f"Columns: {results['dataset_info']['columns']}\n")
                f.write(f"Features Used: {results['dataset_info']['features_used']}\n\n")
                
                f.write("MISSING VALUES (before cleaning):\n")
                for col, count in results['dataset_info']['missing_values'].items():
                    f.write(f"  {col}: {count}\n")
                f.write("\n")
                
                f.write("SPECIES DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                for species, count in results['dataset_info']['species_distribution'].items():
                    f.write(f"  {species}: {count}\n")
                f.write("\n")
                
                f.write("DATA SPLIT:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Training Size: {results['data_split']['train_size']}\n")
                f.write(f"Test Size: {results['data_split']['test_size']}\n")
                f.write(f"Test Ratio: {results['data_split']['test_ratio']}\n\n")
                
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {results['model_performance']['accuracy']:.4f}\n")
                f.write(f"Class Labels: {results['model_performance']['class_labels']}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                conf_matrix = np.array(results['model_performance']['confusion_matrix'])
                class_labels = results['model_performance']['class_labels']
                
                # Print header
                f.write("        Predicted\n")
                f.write("        " + "  ".join(f"{label[:8]:>8}" for label in class_labels) + "\n")
                
                # Print matrix with actual labels
                for i, actual_label in enumerate(class_labels):
                    f.write(f"{actual_label[:8]:>8}")
                    for j in range(len(class_labels)):
                        f.write(f"{conf_matrix[i][j]:>10}")
                    f.write("\n")
                f.write("\n")
                
                f.write("DETAILED CLASSIFICATION REPORT:\n")
                f.write("-" * 40 + "\n")
                f.write(classification_report)
                f.write("\n\n")
                
                f.write("GENERATED FILES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"- Plots saved to: {self.plots_dir}\n")
                f.write(f"- Results saved to: {results_file}\n")
                f.write("- Generated plots:\n")
                f.write("  * svm_confusion_matrix.png\n")
                f.write("  * billlength_distribution.png\n")
                f.write("  * billdepth_distribution.png\n")
                f.write("  * flipperlength_distribution.png\n")
                f.write("  * bodymass_distribution.png\n")
                f.write("  * species_by_island.png\n")
                f.write("  * correlation_matrix.png\n")
                f.write("\n")
                
                f.write("MODEL CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write("- Algorithm: Support Vector Machine (SVM)\n")
                f.write("- Kernel: Linear\n")
                f.write("- Random State: 42\n")
                f.write("- Feature Scaling: StandardScaler\n")
                f.write("- Cross-validation: Stratified train-test split\n")
                
            self.get_logger().info(f"Results saved to: {results_file}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save results to txt file: {e}")


def main(args=None):
    """Main function to initialize and run the ROS2 node"""
    rclpy.init(args=args)
    node = PenguinClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
