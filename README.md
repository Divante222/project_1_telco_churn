# project_1_telco_churn


# Project Description

"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]


# Project Goal

- To identify the driving factors of churn

- To create a model that is capable of predicting which customers are most likely to churn in the future


# Initial Thoughts

- My initial hypothesis is that choosing the correct columns that affect the largest portion of the customer base will yield better results during the modeling phase


# The Plan

- Aquire data from CodeUp Database
    
- Prepare data

    - Created dummy columns for every catagorical object type that could be converted to int type
    - Fix the total_charges column by filling the the spaces with 0 because the people who have a space in their total_charges column have not yet paid their first months bill
    - Convert total_charges column to a float
    - Convert the dummy columns to integers
    - 
    
- Explore the data in search of drivers of churn
    
    - Answer the following questions
        
        - Does having tech support affect churn
        - Does contract type affect churn
        - does payment type affect churn
        - does contrac type affect churn
        
- Develop a Model to predict if a customer will churn
    
    - Use the drivers identified in explore to build a predictive model of different types
    - Evaluate models on train and validate data
    - Search for the best model based on highest accuracy
    - Evaluate the best model on test data
    
- Draw conclusions



================================================================================================
# Data dictionary coming soon
================================================================================================

# Steps to Reproduce

- Clone this repo.

- Acquire the data from CodeUp database using the final_report file

- Run final_report

# Takeaways and Conclusions

- A little above 40% of the people who churned paid by electronic check
- The highest percentage of people who churned had a month to month contract
- The Highest percentage of people who have churned did not have device protection
- The highest percentage of the people who have churned did not have tech support

# Recommendations

- 10 percent off the monthly bill for 3 months for enrolling in any other form of bill payment
- tech support cost reduction for 3 months on 1 year contracts for enrolling in bill payment other than electronic check
- 2 year contract service cost reduction for enrolling in bill payment other than electronic check




     