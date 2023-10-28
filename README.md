# Support Vector Machines (SVM) Project

## Overview

This project is an implementation of Support Vector Machines (SVM) for binary classification. It includes utility functions and examples for SVM-based classification using various kernel functions, both in the dual and primal formulations. Additionally, it provides tools for data visualization, support vector highlighting, and scoring.

## Getting Started

### Prerequisites

Before using this project, ensure you have the following prerequisites installed:

- Python 3.x
- NumPy
- QPSolvers
- Matplotlib

You can install the required packages using `pip`:

```bash
pip install numpy qpsolvers matplotlib
```

### Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/your-svm-project.git
cd your-svm-project
```

2. To use SVM for classification in your own project, you can import the `SVM` class and the utility functions from the provided scripts.

3. You can customize kernel functions, SVM parameters, and data for your specific classification tasks.

4. Run the provided examples to see SVM in action:

- `q1a()`: Demonstrates SVM in the primal form for linear classification.
- `q1b()`: Illustrates SVM in the dual form for linear classification.
- `q2()`: Shows SVM classification using different kernels on a non-linear dataset.
- `q4()`: Evaluates SVM performance with various kernels on the "Wisconsin" dataset.

### Examples

Example usage is provided in the main script `main.py`. You can run the examples there to see how to utilize this SVM project.

```bash
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- This project was inspired by the need for a simple and extensible SVM implementation for educational purposes.
- We thank the authors of the libraries and packages used in this project.

## Contributing

Feel free to contribute to this project by submitting issues, feature requests, or pull requests.

