# HANDLING-OUTLIERS-USING-IQR-STATISTICS
# Notebook: Outlier Detection and Removal using IQR Method

## Overview

This notebook demonstrates a common workflow for detecting and removing outliers from a dataset using the Interquartile Range (IQR) method. Outliers are data points that significantly deviate from the general pattern in a dataset and can negatively impact the performance of machine learning models and statistical analyses.

This notebook guides you through the following steps:

1.  **Importing necessary Python libraries** for data manipulation, numerical operations, and data visualization.
2.  **Creating a sample dataset** with 'Name', 'Height', and 'Weight' features to illustrate outlier detection techniques.
3.  **Detecting potential outliers** in the 'Height' column using:
    *   Histograms to visualize data distribution.
    *   Boxplots for graphical outlier identification.
    *   Percentiles (25th, 50th, 75th) to understand data spread.
    *   IQR (Interquartile Range) calculation to quantify data variability.
    *   Calculation of lower and upper bounds based on IQR for outlier definition.
    *   Identifying data points falling outside these bounds.
4.  **Removing the detected outliers** from the dataset and showcasing the modified dataset.

This notebook provides a practical, step-by-step guide to understanding and applying the IQR method for outlier handling in data analysis.

## Purpose and Motivation

Outliers can arise due to various reasons such as data entry errors, measurement mistakes, or genuinely unusual observations.  It's crucial to identify and handle outliers because they can:

*   **Skew statistical analyses:** Outliers can distort measures of central tendency (like mean) and variability (like standard deviation).
*   **Negatively impact machine learning models:**  Many algorithms are sensitive to outliers, which can lead to biased or less accurate models.
*   **Misrepresent underlying data patterns:** Outliers can obscure the true relationships and trends within the data.

The IQR method is a robust technique for outlier detection because it relies on the interquartile range, which is less sensitive to extreme values compared to methods that use mean and standard deviation. This notebook aims to:

*   **Introduce common techniques for visualizing data distribution (Histograms and Boxplots) to aid in outlier detection.**
*   **Explain the IQR method for defining outlier boundaries.**
*   **Demonstrate how to programmatically identify and remove outliers in Python using Pandas and NumPy.**
*   **Provide a reproducible example for understanding and applying outlier removal techniques.**

## Techniques Demonstrated

This notebook utilizes the following key Python libraries and techniques:

*   **Data Manipulation with `pandas`:** Creating, structuring, and manipulating datasets using Pandas DataFrames.
*   **Numerical Operations with `numpy`:** Performing numerical calculations, particularly for percentile calculation and IQR bounds.
*   **Data Visualization with `seaborn` and `matplotlib.pyplot`:** Creating histograms and boxplots to visually explore data distribution and identify potential outliers.
*   **Percentile Calculation:** Using NumPy to calculate percentiles to understand data distribution and IQR.
*   **Interquartile Range (IQR) Method:**  Applying the IQR method to define lower and upper bounds for outlier detection, a robust statistical technique.
*   **Data Filtering and Removal:**  Using Pandas to filter data based on outlier bounds and remove identified outlier rows from the DataFrame.

## Notebook Structure - Step-by-Step Breakdown

The notebook is structured into the following steps, designed to be followed sequentially:

**1. Import Libraries:**

   - **Code:**
     ```python
     import pandas as pd
     import numpy as np
     import seaborn as sns
     import matplotlib.pyplot as plt
     ```
   - **Description:** This step imports all necessary Python libraries.
     - `pandas`:  For data manipulation using DataFrames.
     - `numpy`: For numerical operations, especially for percentile calculations and array handling.
     - `seaborn`: For creating enhanced and visually appealing statistical plots like histograms and boxplots.
     - `matplotlib.pyplot`:  Provides fundamental plotting functionalities used by Seaborn.
   - **Purpose:** To load the required libraries that provide the tools for data handling, numerical computation, and visualization within the notebook.

** 2. Create Dataset:**

   - **Code:**
     ```python
     Dataset = {'NAME': ['VICKTOR', 'STANLEY', 'KING', 'ALEX', 'ROB', 'TOM', 'KALI'],
                'HEIGHT': [1.2, 4.3, 5.5, 7.2, 5.9, 6.6, 122.5], # Height in feet (FT)
                'WEIGHT': [30, 50, 60, 75, 53, 62, 50]} # Weight in kilograms (KG)
     dataframe = pd.DataFrame(Dataset)
     x = dataframe[['HEIGHT', 'WEIGHT']] # While weight is included, outlier detection focuses on HEIGHT
     y = dataframe['NAME'] # Name is included for reference in the DataFrame, but not directly used for outlier detection
     ```
   - **Description:** This step creates a sample dataset based on individual names, their heights, and weights. This dataset will be used to demonstrate outlier detection using the IQR method, focusing on the 'HEIGHT' column.
     - A dictionary `Dataset` is defined, as containing 'NAME', 'HEIGHT' (in feet - FT), and 'WEIGHT' (in kilograms - KG) for several individuals.  As seen in the dataset, 'KALI' has a significantly larger height (122.5 FT) which is intended to be an outlier for demonstration purposes.
     - `pd.DataFrame(Dataset)`: This converts the dictionary into a Pandas DataFrame named `dataframe` for easier data manipulation and analysis.
     - `x = dataframe[['HEIGHT', 'WEIGHT']]`: Assigns 'HEIGHT' and 'WEIGHT' columns to `x`. While both are included, the outlier detection in this notebook is specifically demonstrated in the 'HEIGHT' column.
     - `y = dataframe['NAME']`: Assigns 'NAME' column to `y`.  'NAME' is included for easier identification of the data points but is not directly used in the outlier detection process itself in this example.
   - **Purpose:** To create a dataset with a clear outlier in the 'HEIGHT' column ('KALI' with 122.5 FT). This dataset allows for a practical demonstration of the IQR method for outlier detection and removal.


   - **Purpose:** To create a dataset that contains a potential outlier in the 'Height' column, allowing for the demonstration of outlier detection and removal techniques.

**3. Outlier Detection:**

   This section is further divided into steps to demonstrate various methods for identifying outliers.

   **a) Histograms:**

      - **Code:**
        ```python
        plt.figure(figsize=(8, 6))
        sns.histplot(dataframe['Height'], bins=10, kde=True)
        plt.xticks(np.arange(dataframe['Height'].min(), dataframe['Height'].max() + 5, 5))
        plt.title('Histogram of Height')
        plt.xlabel('Height (cm)')
        plt.ylabel('Frequency')
        plt.show()
        ```
      - **Description:** Creates a histogram to visualize the distribution of the 'Height' column.
        - `plt.figure(figsize=(8, 6))`: Sets the figure size for better readability.
        - `sns.histplot(dataframe['Height'], bins=10, kde=True)`: Generates a histogram of the 'Height' column with 10 bins and overlays a Kernel Density Estimate (KDE) curve to show the distribution's shape.
        - `plt.xticks(np.arange(dataframe['Height'].min(), dataframe['Height'].max() + 5, 5))`: Sets the x-axis ticks at intervals of 5 cm, starting from the minimum height, for clearer visualization.
        - `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: Sets the plot title and axis labels for clarity.
        - `plt.show()`: Displays the histogram.
      - **Purpose:** To visually inspect the distribution of 'Height' and identify any potential values that stand out as being far from the main cluster of data points. Outliers may appear as isolated bars at the extremes of the histogram.

   **b) Boxplots:**

      - **Code:**
        ```python
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=dataframe['Height'])
        plt.title('Boxplot of Height')
        plt.ylabel('Height (cm)')
        plt.show()
        ```
      - **Description:** Creates a boxplot to graphically represent the distribution of 'Height' and highlight potential outliers.
        - `plt.figure(figsize=(6, 4))`: Sets the figure size.
        - `sns.boxplot(y=dataframe['Height'])`: Generates a boxplot for the 'Height' column. Boxplots visually represent the median, quartiles, and potential outliers (points outside the whiskers).
        - `plt.title(...)`, `plt.ylabel(...)`: Sets the plot title and y-axis label.
        - `plt.show()`: Displays the boxplot.
      - **Purpose:** Boxplots are excellent for visual outlier detection. Outliers are typically represented as points beyond the "whiskers" of the boxplot, providing a quick visual indication of potential extreme values.

   **c) Percentiles:**

      - **Code:**
        ```python
        Q1 = dataframe['Height'].quantile(0.25)
        Q2 = dataframe['Height'].quantile(0.50)
        Q3 = dataframe['Height'].quantile(0.75)
        print(f"25th Percentile (Q1): {Q1}")
        print(f"50th Percentile (Q2 - Median): {Q2}")
        print(f"75th Percentile (Q3): {Q3}")
        ```
      - **Description:** Calculates and prints the 25th (Q1), 50th (Q2 - Median), and 75th (Q3) percentiles of the 'Height' column.
      - **Purpose:** Percentiles provide a numerical understanding of data distribution. Q1, Q2 (median), and Q3 divide the data into quartiles and are essential for IQR calculation. Examining these values gives insights into the spread and central tendency of the data.

   **d) IQR Calculation:**

      - **Code:**
        ```python
        IQR = Q3 - Q1
        print(f"Interquartile Range (IQR): {IQR}")
        ```
      - **Description:** Calculates the Interquartile Range (IQR) as the difference between the 75th percentile (Q3) and the 25th percentile (Q1).
      - **Purpose:** The IQR represents the range of the middle 50% of the data. It is a measure of statistical dispersion and is used to define the outlier bounds in the IQR method.

   **e) Outlier Bounds:**

      - **Code:**
        ```python
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(f"Lower Bound for Outliers: {lower_bound}")
        print(f"Upper Bound for Outliers: {upper_bound}")
        ```
      - **Description:** Calculates the lower and upper bounds for outlier detection using the IQR method.
        - `lower_bound = Q1 - 1.5 * IQR`:  Calculates the lower outlier threshold. Values below this are considered potential outliers.
        - `upper_bound = Q3 + 1.5 * IQR`: Calculates the upper outlier threshold. Values above this are considered potential outliers.
        - The factor 1.5 is a commonly used multiplier in the IQR method, but can be adjusted (e.g., to 3 for more extreme outliers).
      - **Purpose:** These bounds define the range within which data points are considered "normal". Values falling outside these bounds are flagged as potential outliers.

   **f) Finding Outliers:**

      - **Code:**
        ```python
        outliers_height = dataframe[(dataframe['Height'] < lower_bound) | (dataframe['Height'] > upper_bound)]
        outlier_indices_height = outliers_height.index
        print("Identified Height Outliers:\n", outliers_height)
        print("\nIndices of Height Outliers:", outlier_indices_height.tolist())
        ```
      - **Description:** Identifies and stores the rows and indices of data points in the 'Height' column that are considered outliers based on the calculated bounds.
        - `dataframe[(dataframe['Height'] < lower_bound) | (dataframe['Height'] > upper_bound)]`:  This line filters the DataFrame to select rows where the 'Height' value is either less than the `lower_bound` OR greater than the `upper_bound`. This condition identifies the outliers.
        - `outliers_height = ...`: Stores the DataFrame containing only the outlier rows in the `outliers_height` variable.
        - `outlier_indices_height = outliers_height.index`:  Extracts the index labels of the outlier rows and stores them in `outlier_indices_height`.
        - `print(...)`: Prints the identified outlier rows and their indices.
      - **Purpose:** To programmatically pinpoint the exact data points in the 'Height' column that are flagged as outliers according to the IQR method and obtain their row indices for removal in the next step.

**4. Outlier Removal:**

   - **Code:**
     ```python
     dataframe_cleaned = dataframe.drop(outlier_indices_height)
     print("\nDataFrame after Outlier Removal (based on Height):\n", dataframe_cleaned)
     ```
   - **Description:** Removes the identified outlier rows from the original DataFrame to create a cleaned dataset.
     - `dataframe.drop(outlier_indices_height)`: Uses the `.drop()` method of the Pandas DataFrame to remove rows based on the `outlier_indices_height` obtained in the previous step. This creates a new DataFrame without the outlier rows.
     - `dataframe_cleaned = ...`: Assigns the resulting DataFrame (with outliers removed) to the variable `dataframe_cleaned`.
     - `print(...)`: Prints the `dataframe_cleaned` to display the dataset after outlier removal.
   - **Purpose:** To demonstrate the process of removing identified outliers from the dataset, resulting in a modified dataset that is potentially more suitable for further analysis or machine learning tasks.

## How to Use This Notebook

1.  **Prerequisites:** Ensure you have the following Python libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy seaborn matplotlib
    ```

2.  **Open the Notebook:** Open this notebook (e.g., in Jupyter Notebook, JupyterLab, VS Code with Jupyter extension, or Google Colab).

3.  **Run Cells Sequentially:** Execute the notebook cells in order from top to bottom by selecting a cell and pressing `Shift + Enter` (or the "Run" button in your notebook environment).

4.  **Observe the Output:** After running each step, carefully examine the output:
    - **Histograms & Boxplots:** Visually inspect the generated histograms and boxplots to understand the distribution of 'Height' and visually identify potential outliers.
    - **Percentiles and IQR:** Review the printed values for the 25th, 50th, 75th percentiles, and IQR to understand data spread and calculate the outlier bounds.
    - **Outlier Bounds:** Note the calculated lower and upper bounds for outlier detection.
    - **Identified Height Outliers:** Examine the printed `outliers_height` DataFrame and `outlier_indices_height` list to see which rows and indices were identified as outliers based on the IQR method.
    - **DataFrame after Outlier Removal:**  Review the `dataframe_cleaned` output to see the dataset after the identified outliers have been removed. Compare the number of rows in `dataframe_cleaned` to the original `dataframe` to confirm outlier removal.

## Expected Output and Observations

Running the notebook will generate the following outputs:

*   **Histogram of Height:** A histogram visualizing the distribution of the 'HEIGHT' column. You will likely observe that most height values cluster in a certain range, while 'KALI's height (122.5 FT) will appear as a very distant bar on the right side, visually indicating it as a potential outlier.
*   **Boxplot of Height:** A boxplot for 'HEIGHT' will visually highlight potential outliers. 'KALI's height will likely be displayed as a point far beyond the upper whisker of the boxplot, clearly suggesting it as an outlier.
*   **Percentile Values:** Printed values for Q1, Q2 (Median), and Q3 will show the quartiles of the 'HEIGHT' data, excluding the extreme outlier's influence on the quartiles themselves being based on rank.
*   **IQR Value:** The calculated IQR value, representing the spread of the middle 50% of the 'HEIGHT' data, will be printed.
*   **Outlier Bounds:** Printed lower and upper bounds calculated using the IQR method. These bounds will be based on the majority of the height data and will likely flag 'KALI's height as exceeding the upper bound.
*   **Identified Height Outliers:** The `outliers_height` DataFrame will be printed.  This DataFrame will contain the row corresponding to 'KALI' because her 'HEIGHT' value (122.5 FT) falls outside the calculated IQR-based bounds. The `outlier_indices_height` will show the index of 'KALI's row.
*   **DataFrame after Outlier Removal:** The `dataframe_cleaned` DataFrame will be printed, showing the dataset with 'KALI's row removed. You will observe one less row in this data frame compared to the original.



## Key Concepts and Takeaways

By running this notebook, you should gain a better understanding of:

*   **Outliers:** What outliers are and why they are important to consider in data analysis.
*   **Visual Outlier Detection:** Using histograms and boxplots for visually exploring data distribution and identifying potential outliers.
*   **IQR Method for Outlier Detection:**  Understanding and applying the IQR method to define numerical outlier boundaries.
*   **Programmatic Outlier Identification and Removal:**  Learning how to use Python (Pandas and NumPy) to programmatically identify and remove outliers from a dataset.
*   **Impact of Outlier Removal:** Observing how removing outliers can modify a dataset.

## Further Exploration

This notebook provides a basic introduction to outlier detection and removal using the IQR method on a single column ('Height'). You can extend your learning by:

*   **Applying to Other Columns:** Apply the same outlier detection and removal process to the 'weight' column or other numerical columns in the dataset.
*   **Experimenting with IQR Multiplier:** Change the 1.5 multiplier in the IQR bounds calculation (e.g., to 3) to adjust the sensitivity of the outlier detection.  A higher multiplier will make the bounds wider and less sensitive to outliers.
*   **Exploring Other Outlier Detection Methods:** Research and implement other outlier detection techniques, such as Z-score method, DBSCAN (clustering-based outlier detection), or Isolation Forest (machine learning-based outlier detection), and compare their results.
*   **Handling Outliers Instead of Removal:**  Investigate alternative strategies for handling outliers instead of outright removal, such as data transformation (e.g., log transformation) or winsorization (capping extreme values).
*   **Real-World Datasets:** Apply these outlier detection and removal techniques to real-world datasets and observe the impact on data analysis and machine learning model performance.

This notebook serves as a starting point for understanding and handling outliers. By experimenting further, you can develop a more comprehensive understanding and skillset for dealing with outliers in various data analysis scenarios.
