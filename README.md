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


| Features              | Definition                                                                                                         |
|-----------------------|--------------------------------------------------------------------------------------------------------------------|
| customer_id           | Customer ID                                                                                                        |
| gender                | Whether the customer is a male or a female                                                                         |
| senior_citizen        | Whether the customer is a senior citizen or not                                                                    |
| partner               | Whether the customer has a partner or not                                                                          |
| dependents            | Whether the customer has dependents or not                                                                         |
| tenure                | Number of months the customer has stayed with the company                                                          |
| phone_service         | Whether the customer has a phone service or not                                                                    |
| multiple_lines        | Whether the customer has multiple lines or not                                                                     |
| online_security       | Whether the customer has online security or not                                                                    |
| online_backup         | Whether the customer has online backup or not                                                                      |
| device_protection     | Whether the customer has device protection or not                                                                  |
| tech_support          | Whether the customer has tech support or not                                                                       |
| streaming_tv          | Whether the customer has streaming TV or not                                                                       |
| streaming_movies      | Whether the customer has streaming movies or not                                                                   |
| paperless_billing     | Whether the customer has paperless billing or not                                                                  |
| monthly_charges       | The amount charged to the customer monthly                                                                         |
| total_charges         | The total amount charged to the customer                                                                           |
| churn                 | Whether the customer churned or not                                                                                |
| contract_type         | The contract term of the customer (Month-to-month, One year, Two year)                                             |
| internet_service_type | Customer’s internet service provider (DSL, Fiber optic, No)                                                        |
| payment_type          | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |


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




     