Car Price Prediction Model:
This project aims to predict the price of cars in the American market based on various features. The dataset provided by an automobile consulting company contains different attributes of cars, and the goal is to understand the factors affecting car pricing and develop a predictive model.


Project Overview:
The main goal of this project is to build a model that can predict the price of cars using independent variables. The model will be used to understand the pricing dynamics in the American market, enabling the automobile company to adjust their business and production strategies accordingly.

Dataset:
The dataset contains various features of cars, including numeric and categorical columns, such as:

Car dimensions (wheelbase, carlength, carwidth, carheight)
Engine specifications (horsepower, enginesize, boreratio)
Performance metrics (citympg, highwaympg)
Other car-related information (fuel type, aspiration, car body style, etc.)
Dataset Link: Car Price Dataset

Steps and Methodology:

1. Data Preprocessing
Dropped irrelevant columns like car_ID and CarName.
Handled missing values (if any) and removed duplicate records.
Applied label encoding to categorical columns such as doornumber and enginelocation.
One-hot encoded other categorical columns to prepare them for model training.

3. Data Exploration and Visualization
Visualized the distribution of car prices, including applying log transformation for better normalization.
Analyzed the correlations between numeric features using heatmaps.
Detected and capped outliers based on IQR for various numerical features to improve model performance.

5. Feature Engineering
Selected significant features based on their importance using Random Forest.
Performed feature scaling using StandardScaler to ensure all features are on the same scale for certain models.

7. Model Implementation
Five regression algorithms were implemented and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)

5. Model Evaluation
The models were evaluated based on the following metrics:

MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² (R-squared)
The performance metrics for all models were compared, and the best-performing model was selected.

6. Hyperparameter Tuning
The best-performing model, Random Forest Regressor, was tuned using GridSearchCV to find the optimal hyperparameters, improving model accuracy.

7. Final Model Performance
The final model, Logistic Regression, achieved the following performance metrics on the test set:

MAE: 0.1509
MSE: 0.0344
RMSE: 0.1854
R²: 0.8871
The R² score of 0.8871 indicates that the model explains approximately 88.71% of the variance in the car prices, which is a good result.

8. Model Saving and Testing
The final model was saved using joblib for future predictions.
The model was tested on unseen data, and the results were consistent with the training performance.

Conclusion
The final model, Logistic Regression, is highly effective at predicting car prices based on the input features. The results from the evaluation indicate that the model performs well in generalizing to new data, with an R² value of 0.8871. This model can be used by the automobile company to better understand the pricing dynamics of the American car market and make informed decisions regarding car pricing.

Files Included
CarPrice.ipynb: Jupyter notebook containing the full implementation of data preprocessing, model building, evaluation, and hyperparameter tuning.
CarPrice.joblib: Saved final model and scaler for use in future predictions.
