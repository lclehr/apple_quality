# Apple Quality Analysis Project

This project involves analyzing a dataset titled `apple_quality.csv` and processing it using a Python script named `Apple_Quality_git.py`. The analysis aims to predict apple quality and identify distinct apple varieties through various statistical methods including clustering, correlation, and regression analyses as detailed in the provided `CA259_Lehr_Ziegler_Assignment2.pdf`.

## Project Components

### CSV File

- **apple_quality.csv**
  - **Description**: This file contains transformed data about apple quality including attributes like size, weight, sweetness, crunchiness, juiciness, ripeness, and acidity. The data is labeled with quality as "good" or "bad".
  - **Usage**: Used as input for the Python script for analysis. Ensure to handle the data according to its transformed scale which is not specified in the source.

### Python Script

- **Apple_Quality_git.py**
  - **Purpose**: This script processes the `apple_quality.csv` file to perform data cleaning, exploratory data analysis, and machine learning tasks including clustering and regression.
  - **Usage**: Run this script in a Python environment that supports libraries such as Pandas, NumPy, Scikit-Learn, and Matplotlib.

### Documentation

- **CA259_Lehr_Ziegler_Assignment2.pdf**
  - **Purpose**: This document provides a comprehensive analysis of the Apple Quality Dataset including methodologies and findings from various statistical analyses conducted.
  - **Usage**: Review this document to understand the scope of the analyses and the insights drawn from the dataset. It includes details about the data transformation, exploration process, and the results of clustering and regression analyses.

## Getting Started

To get started with analyzing the Apple Quality data:

1. **Environment Setup**:
   - Ensure Python 3.x is installed on your system.
   - Install required Python libraries:
     ```bash
     pip install numpy pandas scikit-learn matplotlib
     ```

2. **Data Preparation**:
   - Download the `apple_quality.csv` file and place it in a known directory.

3. **Run the Analysis**:
   - Execute the `Apple_Quality_git.py` script using Python:
     ```bash
     python Apple_Quality_git.py
     ```
   - The script will output the analysis results, including visualizations and a report on the console.

## Contributing

Feel free to fork this repository or submit pull requests with your suggested changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is provided under the MIT License - see the LICENSE file for details.
