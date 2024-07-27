# Machine Learning Lab

## Worksheet 01

### Exercise 1: Explore Pandas Library

#### Objective
Explore the Pandas library using the dataset provided. Perform operations such as selecting, filtering, aggregating, joining, slicing, and dicing the data to prepare and explore the dataset.

#### Dataset
Download the dataset from the following link: [Dataset](.\IPL.csv)

#### Problems

1. **Read and Load Dataset**:
   - Load the dataset into a Pandas DataFrame.

2. **Use Pandas Features**:
   - **Selecting**:
     - Select specific columns from the DataFrame.
   - **Filtering**:
     - Filter rows based on specific conditions.
   - **Aggregating**:
     - Perform aggregation operations like `sum()`, `mean()`, `count()`, etc.
   - **Joining**:
     - Join two DataFrames using different types of joins (e.g., inner, outer).
   - **Slicing/Dicing**:
     - Slice and dice the data to create subsets of the DataFrame.

3. **Displaying First Few Records**:
   - Display the first few records of the DataFrame using the `head()` method.

4. **Finding Summary of the DataFrame**:
   - Use the `describe()` method to find the summary of the DataFrame.

5. **Slicing and Indexing of DataFrame**:
   - Perform slicing and indexing operations on the DataFrame.

6. **Find Occurrences of Each Unique Value in a Column**:
   - Find the occurrences of each unique value in a specified column.

7. **Perform Cross Tabulation**:
   - Perform cross tabulation between `PLAYING ROLE` and `AGE` to get the number of players in each age category for each playing role.

8. **Sorting DataFrame by Column Values**:
   - Sort the DataFrame by the values in a specified column.

9. **Find Average SOLD PRICE for Each Age Category**:
   - Group all records by `AGE` and then apply the `mean()` method on the `SOLD PRICE` column to find the average SOLD PRICE for each age category.

10. **Perform Joining of DataFrames**:
    - Compare the average SOLD PRICE for different AGE categories with the different age and PLAYING ROLE categories. Merge the DataFrames `soldprice_by_age` and `soldprice_by_age_role` using an outer join on the common column `AGE`. Consider AGE as a categorical variable (e.g., 1, 2, 3).

11. **Calculate Premium for AGE and PLAYING ROLE Categories**:
    - Determine whether players carry a premium if they belong to a specific AGE and PLAYING ROLE category. The premium (change) is calculated in percentage terms.

12. **Exploration of Data Using Visualization**:
    - Use Matplotlib to draw various plots:
        - Bar Chart
        - Scatter Plot
        - Histogram
        - Correlation Plot
        - Heatmap

---
