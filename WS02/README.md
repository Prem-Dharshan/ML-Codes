# Machine Learning Lab

## Worksheet 02 - Statistical Analysis and Regression Exercises

This README provides an overview and guide to the code that addresses three statistical analysis and regression problems. Each problem involves data analysis, linear regression, correlation, and visualization techniques to derive insights and relationships from the given datasets.

## Exercise 2

### Problem 1: Sunshine Hours vs. Ice Cream Sales

**Question:**
Sam found how many hours of sunshine vs. how many ice creams were sold at the shop from Monday to Friday:

| Hours of Sunshine (x) | Ice Creams Sold (y) |
|-----------------------|---------------------|
| 2                     | 4                   |
| 3                     | 5                   |
| 5                     | 7                   |
| 7                     | 10                  |
| 9                     | 15                  |

**Objectives:**
1. Calculate the linear regression equation \( y = mx + b \) using the formula.
2. Compute the correlation coefficient between hours of sunshine and ice cream sales.
3. Find the linear regression line.
4. Determine the least squares of errors.
5. Check the sensitivity of least squares to outliers by introducing an outlier.
6. Draw a box plot to detect outliers and provide an inference on it.

---

### Problem 2: Ice Cream Sales vs. Temperature

**Question:**
The local ice cream shop keeps track of how much ice cream they sell versus the noon temperature on that day. Here are their figures for the last 12 days:

| Temperature (°C) | Ice Cream Sales ($) |
|------------------|---------------------|
| 14.2             | 215                 |
| 16.4             | 325                 |
| 11.9             | 185                 |
| 15.2             | 332                 |
| 18.5             | 406                 |
| 22.1             | 522                 |
| 19.4             | 412                 |
| 25.1             | 614                 |
| 23.4             | 544                 |
| 18.1             | 421                 |
| 22.6             | 445                 |
| 17.2             | 408                 |

**Objectives:**
1. Plot the data points on a scatter plot.
2. Find the line of best fit.
3. Use the regression equation to interpolate the sales value at 21°C.
4. Use the regression equation to extrapolate the sales value at 29°C.

---

### Problem 3: Advertising Data Analysis

**Question:**
Suppose that we are statistical consultants hired by a client to provide advice on how to improve sales of a particular product. The Advertising dataset consists of the sales of that product in 200 different markets, along with advertising budgets for the product in each of those markets for three different media: TV, radio, and newspaper. 

**Dataset:** [Advertising Dataset](./advertising.csv)

Source: [Kaggle](https://www.kaggle.com/datasets/ashydv/advertising-dataset)

| Variable          | Description                                           |
|-------------------|-------------------------------------------------------|
| TV Budget         | Advertising budget for TV in each market ($1000)      |
| Radio Budget      | Advertising budget for Radio in each market ($1000)   |
| Newspaper Budget  | Advertising budget for Newspaper in each market ($1000) |
| Sales             | Sales of the product in each market (units)           |

**Objectives:**
1. Analyze the relationship between sales and TV budget.
2. Check whether increasing sales through advertising has an impact.
3. Construct a confidence interval around the slope of the regression line.

---