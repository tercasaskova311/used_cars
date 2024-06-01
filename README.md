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

# the correlation matrix

----
### Main part(Statistical Methods
1. **2-way ANOVA**:
   - I wanted to determine if there is a relationship between any of the discrete variables         and the price. ANOVA is a technique of the form y=f(x) where y is continuous and x is          discrete.
     We have seen above that, if any, there is a relationship with gearbox, notRepairedDamage,      vehicleType, and FuelType. We will not use brand and monthOfRegistration, as they explain      much less variability than the rest.
     
   - Purpose: Is there a relationship between discrete variables - gear box, not repaired           damage, vehicle type, fuel type and the continuous variable, which is price.
     
   - Findings: Significant interactions were found between some variables. => such as gearbox and not repaired damage (interactions were generally weak). In general, we will not estimate interactions

3. **Multi-way ANOVA**:
   - The R2 of this model is 0.260769

4. **Pure Regression**:
   - Linear regression was used to model the price of the car based on continuous variables.
   - The average price of a car is 5856.7, for every year the car loses 473 euros. More             powerful cars are more expensive (46 euros/Power horse). And cars with more kilometers         are cheaper, -0.04 euros/km.
   - Results:
     - The average price of a car was found to be €5856.7.
     - The price decreases by €473 for every additional year of the car's age.
     - More powerful cars (measured in horsepower) are more expensive by €46 per additional           horsepower, while higher mileage decreases the price by €0.04 per kilometer.
    
     - Notes:
      [1] Standard Errors assume that the covariance matrix of the errors is correctly
      specified.
      [2] The condition number is large, 2.43e+06. This might indicate that there are
      strong multicollinearity or other numerical problems.
      The R2 of this model is 0.729726

5. **Mixed Models**:
   - Included both fixed effects (e.g., year, horsepower) and random effects to account for variability not captured by the fixed effects alone.
   - Quadratic terms were considered to explore non-linear relationships between variables and price.

### Results:
-----
- The analysis findings showed that certain variables significantly impact the price of used cars, though some interactions were weak.
- The regression model provided a clear quantification of how different factors such as age, power, and mileage affect car prices.
- By incorporating mixed models and quadratic terms, the analysis aimed to refine the understanding of these relationships, although the primary findings indicated the dominant effects were captured by the simpler models.

- The use of ANOVA and regression techniques highlighted key relationships and provided actionable insights into how various attributes of cars affect their market value. The findings can inform both buyers and sellers in the used car market, helping to set realistic price expectations based on car characteristics.
