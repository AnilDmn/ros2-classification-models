import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from ament_index_python.packages import get_package_share_directory

class FruitClassifierNode(Node):
    def __init__(self):
        super().__init__('fruit_classifier_node')
        self.publisher_ = self.create_publisher(String, 'classification_result', 10)
        
        # Set output directory to the specified WSL path
        self.output_dir = "/home/anild/ros2_ml_ws/src/ml_classifiers_pkg"
        self.plots_dir = os.path.join(self.output_dir, 'plots', 'fruits')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.get_logger().info("Fruit Classifier Node started.")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.classify_fruits()

    def classify_fruits(self):
        """Main classification pipeline following the assignment requirements"""
        try:
            # Step 1: Load dataset into ROS environment
            package_share_directory = get_package_share_directory('ml_classifiers_pkg')
            csv_path = os.path.join(package_share_directory, 'data', 'fruits_weight_sphercity.csv')
            df = pd.read_csv(csv_path)
            self.get_logger().info("CSV loaded successfully.")
            self.get_logger().info(f"Dataset shape: {df.shape}")
            self.get_logger().info(f"Dataset columns: {list(df.columns)}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV file: {e}")
            return

        # Dictionary to store results
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns)
            }
        }

        # Step 1: Feature Engineering and Label Extraction
        self.get_logger().info("Starting Feature Engineering...")
        
        # Identify features and labels
        available_features = [col for col in df.columns if col.lower() != 'labels']
        self.get_logger().info(f"Available features: {available_features}")
        
        # Check if Color column exists and encode it if present
        if 'Color' in df.columns:
            # Encode categorical color variables into numerical values
            color_mapping = {'Green': 20, 'Red': 80, 'Yellow': 50, 'Orange': 60, 'Blue': 30, 'Purple': 40, 'Greenish yellow': 55, 'Reddish yellow': 70}
            df['Color_Encoded'] = df['Color'].map(color_mapping)
            if df['Color_Encoded'].isnull().any():
                self.get_logger().warning("Some colors couldn't be mapped, using label encoding")
                # Use label encoding for unmapped colors
                le = LabelEncoder()
                df['Color_Encoded'] = le.fit_transform(df['Color'])
            features = ['Weight', 'Sphericity', 'Color_Encoded']
        else:
            # Use available numerical features
            features = ['Weight', 'Sphericity']
        
        # Encode class labels (dependent variables)
        if 'labels' in df.columns:
            label_col = 'labels'
        elif 'Label' in df.columns:
            label_col = 'Label'
        else:
            self.get_logger().error("No label column found")
            return
            
        # Map fruit names to numerical values
        df['Label_Encoded'] = df[label_col].map({'apple': 0, 'orange': 1, 'Apple': 0, 'Orange': 1})
        if df['Label_Encoded'].isnull().any():
            self.get_logger().error("Invalid class labels found in CSV.")
            return

        # Select features (independent variables) and labels (dependent variables)
        X = df[features]
        y = df['Label_Encoded']
        
        self.get_logger().info(f"Using features: {features}")
        self.get_logger().info(f"Feature shapes - X: {X.shape}, y: {y.shape}")
        
        results['features'] = features
        results['feature_shapes'] = {'X': X.shape, 'y': y.shape}
        
        # Feature engineering: Normalize/standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.get_logger().info("Features normalized using StandardScaler")

        # Step 2: Data Visualization
        self.get_logger().info("Creating data visualizations...")
        self.visualize_data_distribution(df, features, label_col)

        # Step 3: Data Splitting
        # Split dataset into training and testing sets (70/30 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        self.get_logger().info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        results['data_split'] = {'train_size': X_train.shape[0], 'test_size': X_test.shape[0]}

        # Step 4: Model Fitting with Cross-Validation
        # Fit a Linear Classifier (Logistic Regression) to training data
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        self.get_logger().info(f"Cross-validation scores: {cv_scores}")
        self.get_logger().info(f"CV Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results['cross_validation'] = {
            'scores': cv_scores.tolist(),
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        model.fit(X_train, y_train)
        self.get_logger().info("Linear classifier (Logistic Regression) fitted to training data")

        # Step 5: Model Prediction and Evaluation
        # Make predictions on test dataset
        y_pred = model.predict(X_test)

        # Evaluate model performance using multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=['Apple', 'Orange'])
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Add results to dictionary
        results['model_performance'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }

        # Document findings
        self.get_logger().info("=== MODEL EVALUATION RESULTS ===")
        self.get_logger().info(f"Model accuracy: {accuracy:.4f}")
        self.get_logger().info(f"Precision: {precision:.4f}")
        self.get_logger().info(f"Recall: {recall:.4f}")
        self.get_logger().info(f"F1-Score: {f1:.4f}")
        self.get_logger().info("Classification report:\n" + report)

        # Publish results to ROS topic
        result_msg = String()
        result_msg.data = f"Fruit Classification - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, CV Mean: {cv_scores.mean():.4f}"
        self.publisher_.publish(result_msg)

        # Save visualizations and confusion matrix
        self.save_model_results(df, conf_matrix, features, label_col)
        
        # Save results to txt file
        self.save_results_to_txt(results, report)

    def visualize_data_distribution(self, df, features, label_col):
        """Create visualizations to understand data structure and relationships"""
        
        # Plot relationship between Weight and Sphericity (main requirement)
        plt.figure(figsize=(10, 6))
        colors = {'apple': 'red', 'orange': 'orange', 'Apple': 'red', 'Orange': 'orange'}
        
        for fruit_type in df[label_col].unique():
            subset = df[df[label_col] == fruit_type]
            plt.scatter(subset['Weight'], subset['Sphericity'], 
                       label=fruit_type.capitalize(), 
                       color=colors.get(fruit_type, 'blue'), 
                       s=100, alpha=0.7)
        
        plt.title("Fruit Dataset: Weight vs. Sphericity Distribution")
        plt.xlabel("Weight (grams)")
        plt.ylabel("Sphericity")
        plt.legend(title="Fruit Type")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(self.plots_dir, 'weight_vs_sphericity.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f"Weight vs Sphericity plot saved to: {scatter_path}")

        # Additional visualization: Feature distributions
        if len(features) > 2:
            fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
            if len(features) == 1:
                axes = [axes]
            for i, feature in enumerate(features):
                for fruit_type in df[label_col].unique():
                    subset = df[df[label_col] == fruit_type]
                    axes[i].hist(subset[feature], alpha=0.6, label=fruit_type.capitalize(), bins=10)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            dist_path = os.path.join(self.plots_dir, 'feature_distributions.png')
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Feature distributions plot saved to: {dist_path}")

    def save_model_results(self, df, conf_matrix, features, label_col):
        """Save confusion matrix and other result visualizations"""
        sns.set(style="whitegrid")

        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                    xticklabels=['Apple', 'Orange'],
                    yticklabels=['Apple', 'Orange'],
                    cbar_kws={'label': 'Count'})
        plt.title("Confusion Matrix - Fruit Classification")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        conf_path = os.path.join(self.plots_dir, 'confusion_matrix.png')
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f"Confusion matrix saved to: {conf_path}")

    def save_results_to_txt(self, results, classification_report):
        """Save results to .txt files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'fruit_classification_results_{timestamp}.txt')
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FRUIT CLASSIFICATION RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Dataset Shape: {results['dataset_info']['shape']}\n")
                f.write(f"Columns: {results['dataset_info']['columns']}\n")
                f.write(f"Features Used: {results['features']}\n")
                f.write(f"Feature Shapes: {results['feature_shapes']}\n\n")
                
                f.write("DATA SPLIT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Training Size: {results['data_split']['train_size']}\n")
                f.write(f"Test Size: {results['data_split']['test_size']}\n\n")
                
                f.write("CROSS-VALIDATION RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"CV Scores: {results['cross_validation']['scores']}\n")
                f.write(f"CV Mean: {results['cross_validation']['mean']:.4f}\n")
                f.write(f"CV Std: {results['cross_validation']['std']:.4f}\n\n")
                
                f.write("MODEL PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['model_performance']['accuracy']:.4f}\n")
                f.write(f"Precision: {results['model_performance']['precision']:.4f}\n")
                f.write(f"Recall: {results['model_performance']['recall']:.4f}\n")
                f.write(f"F1-Score: {results['model_performance']['f1_score']:.4f}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 30 + "\n")
                conf_matrix = np.array(results['model_performance']['confusion_matrix'])
                f.write("     Predicted\n")
                f.write("     Apple  Orange\n")
                f.write(f"Apple  {conf_matrix[0][0]:3d}    {conf_matrix[0][1]:3d}\n")
                f.write(f"Orange {conf_matrix[1][0]:3d}    {conf_matrix[1][1]:3d}\n\n")
                
                f.write("DETAILED CLASSIFICATION REPORT:\n")
                f.write("-" * 30 + "\n")
                f.write(classification_report)
                f.write("\n\n")
                
                f.write("FILES GENERATED:\n")
                f.write("-" * 30 + "\n")
                f.write(f"- Plots saved to: {self.plots_dir}\n")
                f.write(f"- Results saved to: {results_file}\n")
                f.write("- Generated plots:\n")
                f.write("  * weight_vs_sphericity.png\n")
                f.write("  * confusion_matrix.png\n")
                if len(results['features']) > 2:
                    f.write("  * feature_distributions.png\n")
                
            self.get_logger().info(f"Results saved to: {results_file}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save results to txt file: {e}")

def main(args=None):
    """Main function to initialize and run the ROS2 node"""
    rclpy.init(args=args)
    node = FruitClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
