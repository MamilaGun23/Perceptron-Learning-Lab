# Perceptron Algorithm Implementation

A hands-on implementation of the Perceptron learning algorithm from scratch, with applications to synthetic data and the Iris dataset. This project demonstrates fundamental concepts of linear classifiers and machine learning.

## 📁 Project Structure

```
├── perceptron_implementation.py   # Main Python script with all exercises
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## 🎯 Overview

The Perceptron is one of the simplest artificial neural networks and a fundamental building block of machine learning. This project implements the Perceptron algorithm from scratch and explores its behavior on different datasets.

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/perceptron-implementation.git
cd perceptron-implementation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy matplotlib scikit-learn
```

## 📈 Usage

Run the main script:
```bash
python perceptron_implementation.py
```

The script executes five comprehensive exercises:

### Exercise 1: Basic Perceptron Implementation
- Implements a Perceptron class from scratch
- Includes `fit()`, `predict()`, and `net_input()` methods
- Tests on a simple book classification dataset (size, color → fiction/non-fiction)

### Exercise 2: Learning Progress Visualization
- Plots errors per epoch to visualize learning convergence
- Analyzes how the Perceptron learns over time

### Exercise 3: Decision Boundary Visualization
- Creates a 2D visualization of the decision boundary
- Plots training data and predictions for new samples
- Shows how the Perceptron separates classes in feature space

### Exercise 4: Hyperparameter Tuning
- Compares different learning rates (η) and iteration counts
- Analyzes the impact of η=0.01 vs η=0.5 on learning speed and stability
- Demonstrates convergence behavior with different parameters

### Exercise 5: Real-world Application - Iris Dataset
- Applies the Perceptron to the classic Iris dataset
- Uses petal length and width to distinguish Setosa from Versicolor
- Visualizes the decision boundary on real biological data

## 📊 Results

### Synthetic Dataset (Book Classification):
- **Final Accuracy**: 100% (linearly separable)
- **Convergence**: Achieved by epoch 7
- **New Sample Prediction**: [3, 2] → Non-fiction (-1)

### Iris Dataset (Flower Classification):
- **Final Accuracy**: 100% (Setosa vs Versicolor)
- **Convergence**: Achieved by epoch 3
- **New Sample Prediction**: [4.0, 1.0] → Versicolor (1)

## 🤔 Conceptual Insights

### Key Learnings:

1. **Linearly Separable Data**: The Perceptron only converges if classes can be separated by a straight line/hyperplane
2. **Learning Rate (η) Impact**:
   - η=0.01: Slow, stable convergence
   - η=0.5: Fast but potentially oscillatory convergence
3. **Error Analysis**: Errors decrease over epochs as the decision boundary improves
4. **Decision Boundary**: A straight line in 2D space that separates classes

### Perceptron Characteristics:
- **Online Learning**: Updates weights after each sample
- **Binary Classification**: Outputs +1 or -1
- **Linear Classifier**: Cannot learn non-linear decision boundaries
- **Convergence**: Guaranteed for linearly separable data (Perceptron Convergence Theorem)

## 📋 Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Scikit-learn (for Iris dataset only)

## 📝 Code Example

```python
# Basic Perceptron usage
from perceptron import Perceptron

# Create synthetic data
X = np.array([[2, 3], [1, 1], [4, 5]])
y = np.array([1, -1, 1])

# Train model
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)

# Make prediction
prediction = model.predict(np.array([3, 2]))
print(f"Prediction: {prediction}")
```

## 🎓 Educational Value

This project is ideal for:
- Understanding the fundamentals of neural networks
- Learning about linear classifiers and decision boundaries
- Exploring hyperparameter tuning (learning rate, iterations)
- Visualizing machine learning concepts
- Comparing synthetic and real-world datasets

## 🔍 Limitations and Extensions

### Current Limitations:
1. Only works for linearly separable data
2. Binary classification only
3. No regularization or advanced features

### Possible Extensions:
1. Add multi-class Perceptron (One-vs-All)
2. Implement Pocket Algorithm for non-separable data
3. Add regularization terms
4. Extend to multi-layer Perceptron

## 📚 Resources

- [Perceptron Convergence Theorem Proof](https://en.wikipedia.org/wiki/Perceptron)
- [Iris Dataset Documentation](https://archive.ics.uci.edu/ml/datasets/iris)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:
- Additional visualization options
- More datasets for testing
- Performance optimizations
- Educational explanations

## 📝 License

This project is open source and available under the MIT License.

---

**Note**: The Perceptron is a historical algorithm that laid the foundation for modern neural networks. While simple, it demonstrates core concepts essential for understanding more complex models.
