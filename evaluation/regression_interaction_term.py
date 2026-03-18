import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


def run_regression_for_fairness_measures(data, model_based, dataset_based, path, run_random_forest=False):
    fairness_measures_selected = [
        'Value Unfairness of sensitive attribute',
        'Overestimation Unfairness of sensitive attribute',
        'Differential Fairness of sensitive attribute',
        'Generalized Cross Entropy',
        'KS Statistic of sensitive attribute'
    ]
    dropped_accuracy_metrics = ["recall@5", "ndcg@5", "mrr@5"]

    for measure in fairness_measures_selected:
        print(f"\n--- Running Regression for {measure} ---")

        data_temp = data[data["Is Filtered"] == "Yes"]

        if (model_based==False) & (dataset_based==False):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID'])
        categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]

        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + ["Is Filtered"]]

        # One-hot encode all categorical variables
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(data_temp[categorical_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

        # Combine encoded features with numeric features
        X_numeric = data_temp[numeric_features].astype(float)
        X_preprocessed = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

        #INTERACTION TERMS ARE NOT USED IN THIS VERSION

        # IN ORDER TO USE INTERACTION TERMS, UNCOMMENT THE FOLLOWING CODE,
        # AND COMMENT THE PREVIOUS CODE

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_preprocessed)
        # Get the feature names for interaction terms
        interaction_feature_names = poly.get_feature_names_out(X_preprocessed.columns)
        # Create DataFrame with interaction features
        X_poly_df = pd.DataFrame(X_poly, columns=interaction_feature_names)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_poly_df)

        # Add a constant for the OLS regression
        X_with_constant = sm.add_constant(X_scaled)
        features_with_const = list(interaction_feature_names) # ONLY FOR INTERACTION TERMS
        #features_with_const = ["const"] + list(X_preprocessed.columns)
        y = data.loc[data_temp.index, measure]

        ols_model = sm.OLS(y, X_scaled).fit()

        coefficients = ols_model.params.values
        p_values = ols_model.pvalues.values

        # Adjust lengths to match
        min_length = min(len(features_with_const), len(coefficients), len(p_values))
        features_with_const = features_with_const[:min_length]
        coefficients = coefficients[:min_length]
        p_values = p_values[:min_length]

        # Add significance stars
        significance = []
        for p in p_values:
            if p <= 0.001:
                significance.append("***")
            elif p <= 0.01:
                significance.append("**")
            elif p <= 0.05:
                significance.append("*")
            else:
                significance.append("")

        # Create a DataFrame with the results
        importance_df = pd.DataFrame({
            "Feature": features_with_const,
            "Coefficient": coefficients,
            "P-Value": p_values,
            "Significance": significance
        }).sort_values(by="Coefficient", ascending=False)

        # Save the results to an Excel file for each measure
        output_file = f"./{path}/OLS_Regression_Feature_Analysis_{measure.replace(' ', '_')}_interaction.xlsx"
        importance_df.to_excel(output_file, index=False)

        # Print R² scores and summary
        ols_summary = ols_model.summary()
        print("R² Score:", ols_model.rsquared)
        print("Adjusted R² Score:", ols_model.rsquared_adj)
        print(ols_summary)

        if run_random_forest:
            rf_model = RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            )
            rf_model.fit(X_preprocessed, y)
            rf_r2 = rf_model.score(X_preprocessed, y)
            print("RF R² Score (train):", rf_r2)

            rf_importance_df = pd.DataFrame({
                "Feature": X_preprocessed.columns,
                "Importance": rf_model.feature_importances_,
            }).sort_values(by="Importance", ascending=False)

            rf_output_file = f"./{path}/RF_Feature_Importance_{measure.replace(' ', '_')}.xlsx"
            rf_importance_df.to_excel(rf_output_file, index=False)

def concat_regression_results(model_based, dataset_based, path, model_name, dataset_name, sensitive_feature):

    # List of uploaded files and their corresponding target measures
    files = {
        "Value Unfairness": "OLS_Regression_Feature_Analysis_Value_Unfairness_of_sensitive_attribute_interaction.xlsx",
        "Overestimation Unfairness": "OLS_Regression_Feature_Analysis_Overestimation_Unfairness_of_sensitive_attribute_interaction.xlsx",
        "Differential Fairness": "OLS_Regression_Feature_Analysis_Differential_Fairness_of_sensitive_attribute_interaction.xlsx",
        "Generalized Cross Entropy": "OLS_Regression_Feature_Analysis_Generalized_Cross_Entropy_interaction.xlsx",
        'KS Statistic': "OLS_Regression_Feature_Analysis_KS_Statistic_of_sensitive_attribute_interaction.xlsx",
    }
    # Initialize an empty list to store dataframes
    dfs = []

    # Read each file and add a "Target Measure" column
    for target, file in files.items():
        df = pd.read_excel(f"./{path}/"+file)
        df["Target Measure"] = target
        dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)
    output_file = ""
    if (model_based == False) & (dataset_based == False):
        # Save the concatenated dataframe to a single Excel file
        output_file = f"RQ3_OLS_Regression_Feature_Analysis_fairness_measure_based_interaction.xlsx"

    concatenated_df.to_excel(f"./{path}/"+output_file, index=False)



dropped_fairness_measures = [
    'Absolute Unfairness of sensitive attribute',
    'Underestimation Unfairness of sensitive attribute',
    'NonParity Unfairness of sensitive attribute',
    'Absolute Difference',
    #'Generalized Cross Entropy',
    'giniindex@5', "popularitypercentage@5"
    ]
dropped_dc_08 = ['Space Size', 'Average Popularity', 'Standart Deviation of Popularity Bias', 'Kurtosis of Popularity Bias', 'Average Long Tail Items', 'Standart Deviation of Long Tail Items', 'Kurtosis of Long Tail Items', 'Skewness of Rating', 'Kurtosis of Rating']
dropped_dc_07 = ['Number of Ratings', 'Space Size', 'Rating per User', 'Rating per Item', 'Gini Item', 'Gini User', 'Average Popularity', 'Standart Deviation of Popularity Bias', 'Skewness of Popularity Bias', 'Kurtosis of Popularity Bias', 'Average Long Tail Items', 'Standart Deviation of Long Tail Items', 'Skewness of Long Tail Items', 'Kurtosis of Long Tail Items', 'Mean Rating', 'Standart Deviation of Rating', 'Skewness of Rating', 'Kurtosis of Rating']

model_based = False
run_random_forest = False
dataset_based = False


data = pd.read_csv("df_regression.csv", index_col=0)
data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
run_regression_for_fairness_measures(
    data,
    model_based,
    dataset_based,
    "fairness_measure_based",
    run_random_forest=run_random_forest,
)
concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based",
                          model_name=None, dataset_name=None, sensitive_feature=None)

print("RQ3 DONE")
