### Used cars data set (the price of the cars vs factors).

----
### Project Summary
This project involves analyzing a dataset of used cars to determine the relationship between various factors and the price of the cars. The analysis includes data preparation, exploration, and the application of several statistical methods.

My main focus with this projectc was on ANOVA, multi-way ANOVA, regression, and mixed models.
----
### Data Preparation
The dataset was obtained from a Kaggle repository containing 50,000 data points of used cars. I used the following data set: https://www.kaggle.com/datasets/piumiu/used-cars-database-50000-data-points

Initial steps included loading the data, removing rows with empty values, and ensuring the data types of the columns were appropriate for analysis.

### Data Exploration
The exploration phase involved filtering the dataset to focus on specific years and brands of interest. I did this part to get relevant data points.
----
### Main part(Statistical Methods
1. **Two-way ANOVA**:
   - Purpose: Is there a relationship between discrete variables - gear box, not repaired damage, vehicle type, fuel type and the continuous variable, which is price.
   - Findings: Significant interactions were found between some variables.
      => such as gearbox and not repaired damage (interactions were generally weak).

2. **Multi-way ANOVA**:
   - I have used two-way ANOVA to include more factors and their interactions = to understand the combined effects on the price.

3. ** Regression**:
   - Linear regression was used to model the price of the car based on continuous variables.
   - Results:
     - The average price of a car was found to be €5856.7.
     - The price decreases by €473 for every additional year of the car's age.
     - More powerful cars (measured in horsepower) are more expensive by €46 per additional horsepower, while higher mileage decreases the price by €0.04 per kilometer.

4. **Mixed Models**:
   - Included both fixed effects (e.g., year, horsepower) and random effects to account for variability not captured by the fixed effects alone.
   - Quadratic terms were considered to explore non-linear relationships between variables and price.

### Results:
-----
- The analysis findings showed that certain variables significantly impact the price of used cars, though some interactions were weak.
- The regression model provided a clear quantification of how different factors such as age, power, and mileage affect car prices.
- By incorporating mixed models and quadratic terms, the analysis aimed to refine the understanding of these relationships, although the primary findings indicated the dominant effects were captured by the simpler models.

- The use of ANOVA and regression techniques highlighted key relationships and provided actionable insights into how various attributes of cars affect their market value. The findings can inform both buyers and sellers in the used car market, helping to set realistic price expectations based on car characteristics.
