import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import shap
import numpy as np

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def run_regression_for_fairness_measures2(
    data, model_based, dataset_based, path, run_random_forest=False,
    correction_scope="RQ1",   # sadece açıklama amaçlı label
    correction_method="bonferroni",  # "bonferroni" or "fdr_bh"
    alpha=0.05
):
    fairness_measures_selected = [
        'Value Unfairness of sensitive attribute',
        'Overestimation Unfairness of sensitive attribute',
        'Differential Fairness of sensitive attribute',
        'Generalized Cross Entropy',
        'KS Statistic of sensitive attribute'
    ]
    dropped_accuracy_metrics = ["recall@5", "ndcg@5", "mrr@5"]

    # ✅ 1) Tüm regresyonlardan çıkan katsayı satırlarını burada biriktireceğiz
    all_rows = []

    for measure in fairness_measures_selected:
        print(f"\n--- Running Regression for {measure} ---")

        data_temp = data[data["Is Filtered"] == "Yes"].copy()

        if (model_based==True) & (dataset_based==False):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID', 'Model Name'],
                errors="ignore"
            )
            categorical_features = ["Sensitive Feature", "Dataset"]

        elif (model_based==False) & (dataset_based==True):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID',"Sensitive Feature", "Dataset"],
                errors="ignore"
            )
            categorical_features = ["Model Name"]

        else:
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID'],
                errors="ignore"
            )
            categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]

        numeric_features = [c for c in data_temp.columns if c not in categorical_features]

        # One-hot
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(data_temp[categorical_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

        # Numeric
        X_numeric = data_temp[numeric_features].astype(float).reset_index(drop=True)
        X_encoded_df = X_encoded_df.reset_index(drop=True)
        X_preprocessed = pd.concat([X_numeric, X_encoded_df], axis=1)

        # Scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_preprocessed)

        # ✅ 2) statsmodels’e DataFrame ver → isim/sıra garanti
        X_df = pd.DataFrame(X_scaled, columns=X_preprocessed.columns)
        X_df = sm.add_constant(X_df, has_constant="add")  # const garanti

        # ✅ 3) y'yi doğru yerden al (data_temp)
        y = data[data["Is Filtered"] == "Yes"][measure].astype(float).reset_index(drop=True)
        y = (y - y.mean()) / y.std()
        ols_model = sm.OLS(y, X_df).fit()

        # Raw outputs (Series with index)
        coef = ols_model.params
        pval = ols_model.pvalues
        tval = ols_model.tvalues

        df_resid = float(ols_model.df_resid)
        r2_full = float(ols_model.rsquared)

        # Partial R², Cohen f² (const dahil; istersen const’u sonra drop edersin)
        partial_r2 = (tval.values ** 2) / ((tval.values ** 2) + df_resid)
        den = max(1e-12, (1.0 - r2_full))
        cohen_f2 = partial_r2 / den

        tmp = pd.DataFrame({
            "Target Measure": measure,
            "Feature": coef.index,
            "Coefficient": coef.values,
            "t": tval.values,
            "P-Value (raw)": pval.values,
            "Partial_R2": partial_r2,
            "Cohen_f2": cohen_f2,
        })

        # ✅ const’u correction havuzuna sokmamak için flag ekle
        tmp["Is_Const"] = tmp["Feature"].eq("const")

        all_rows.append(tmp)

    # ✅ 4) Hepsini birleştir
    df_all = pd.concat(all_rows, ignore_index=True)

    # =========================
    # GLOBAL MULTIPLE TESTING CORRECTION (Reviewer’ın istediği)
    # =========================
    # Havuz: correction_scope boyunca tüm katsayı testleri (const hariç)
    mask = ~df_all["Is_Const"]
    p_pool = df_all.loc[mask, "P-Value (raw)"].astype(float).values

    if correction_method == "bonferroni":
        m = len(p_pool)  # ✅ artık m = "tüm regresyon analizleri boyunca yapılan test sayısı"
        p_adj = np.minimum(p_pool * m, 1.0)

    elif correction_method == "fdr_bh":
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(p_pool, alpha=alpha, method="fdr_bh")

    else:
        raise ValueError("correction_method must be 'bonferroni' or 'fdr_bh'")

    # Yaz geri
    df_all["P-Value (corrected)"] = np.nan
    df_all.loc[mask, "P-Value (corrected)"] = p_adj

    # Significance kolonları (raw + corrected)
    def stars(p):
        if pd.isna(p): return ""
        if p <= 0.001: return "***"
        if p <= 0.01:  return "**"
        if p <= 0.05:  return "*"
        return ""

    df_all["Significance (raw)"] = df_all["P-Value (raw)"].apply(stars)
    df_all["Significance (corrected)"] = df_all["P-Value (corrected)"].apply(stars)

    # =========================
    # İstersen measure bazında ayrı Excel yaz
    # =========================
    for measure in df_all["Target Measure"].unique():
        out = df_all[df_all["Target Measure"] == measure].copy()
        out = out.sort_values("Partial_R2", ascending=False)

        output_file = f"./{path}/OLS_Regression_Feature_Analysis_{measure.replace(' ', '_')}_{correction_method}_{correction_scope}.xlsx"
        out.to_excel(output_file, index=False)

    # Toplu output da kaydet (RQ1 tablolar için en kullanışlısı)
    output_all = f"./{path}/RQ1_AllMeasures_OLS_Results_{correction_method}_{correction_scope}.xlsx"
    df_all.to_excel(output_all, index=False)
    print("Saved:", output_all)

    return df_all

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

        if (model_based==True) & (dataset_based==False):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID', 'Model Name'])
            categorical_features = ["Sensitive Feature", "Dataset"]
        elif (model_based==False) & (dataset_based==True):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID',"Sensitive Feature", "Dataset"])
            categorical_features = ["Model Name"]
        elif (model_based==False) & (dataset_based==False):
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics +fairness_measures_selected+
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID'])
            categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]

        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + ["Is Filtered"]]

        # One-hot encode all categorical variables
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(data_temp[categorical_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

        #encoder = ce.BinaryEncoder(cols=categorical_features)
        #X_encoded_df = encoder.fit_transform(data_temp[categorical_features])

        # Combine encoded features with numeric features
        X_numeric = data_temp[numeric_features].astype(float)
        X_preprocessed = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

        #INTERACTION TERMS ARE NOT USED IN THIS VERSION

        # IN ORDER TO USE INTERACTION TERMS, UNCOMMENT THE FOLLOWING CODE, 
        # AND COMMENT THE PREVIOUS CODE

        #poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        #X_poly = poly.fit_transform(X_preprocessed)
        # Get the feature names for interaction terms
        #interaction_feature_names = poly.get_feature_names_out(X_preprocessed.columns)
        # Create DataFrame with interaction features
        #X_poly_df = pd.DataFrame(X_poly, columns=interaction_feature_names)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_preprocessed) # change to X_poly

        # Add a constant for the OLS regression
        X_with_constant = sm.add_constant(X_scaled)
        #features_with_const =  ["const"] + list(interaction_feature_names)
        features_with_const = ["const"] + list(X_preprocessed.columns)
        y = data[measure]
        y = (y - y.mean()) / y.std(ddof=0)
        ols_model = sm.OLS(y, X_with_constant).fit()

        coefficients = ols_model.params.values
        p_values = ols_model.pvalues.values

        # NEW: t-stats and df_resid
        t_values = ols_model.tvalues.values
        df_resid = float(ols_model.df_resid)
        r2_full = float(ols_model.rsquared)

        # Compute Partial R^2 for each coefficient using t-stat
        partial_r2 = (t_values ** 2) / ((t_values ** 2) + df_resid)

        # Compute Cohen's f^2 per feature
        den = max(1e-12, (1.0 - r2_full))  # avoid division by zero
        cohen_f2 = partial_r2 / den

        pvals = np.array(p_values, dtype=float)
        mask = np.array([f != "const" for f in features_with_const])

        m = int(mask.sum())  # kaç test var (const hariç)

        pvals_bonf = np.full_like(pvals, fill_value=np.nan, dtype=float)
        pvals_bonf[mask] = np.minimum(pvals[mask] * m, 1.0)  # p*m, 1'i aşarsa kırp

        min_length = min(
            len(features_with_const),
            len(coefficients),
            len(p_values),
            len(t_values),
            len(partial_r2),
            len(cohen_f2)
        )

        features_with_const = features_with_const[:min_length]
        coefficients = coefficients[:min_length]
        p_values = p_values[:min_length]
        t_values = t_values[:min_length]
        partial_r2 = partial_r2[:min_length]
        cohen_f2 = cohen_f2[:min_length]

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

        importance_df = pd.DataFrame({
            "Feature": features_with_const,
            "Coefficient": coefficients,
            "t": t_values,
            "P-Value (raw)": pvals,
            "P-Value (Bonferroni)": pvals_bonf,
            "Reject (Bonf@0.05)": np.where(mask, pvals_bonf <= 0.05, False),
            "Partial_R2": partial_r2,
            "Cohen_f2": cohen_f2,
        })

        def stars(p):
            if np.isnan(p): return ""
            if p <= 0.001: return "***"
            if p <= 0.01:  return "**"
            if p <= 0.05:  return "*"
            return ""

        importance_df["Significance (raw)"] = [stars(p) for p in importance_df["P-Value (raw)"]]
        importance_df["Significance (bonf)"] = [stars(p) for p in importance_df["P-Value (Bonferroni)"]]

        # (Optional) Drop constant row from the exported table
        # importance_df = importance_df[importance_df["Feature"] != "const"]

        # Sort by Partial R² (more meaningful than raw coefficient)
        importance_df = importance_df.sort_values(by="Partial_R2", ascending=False)

        # Save the results to an Excel file for each measure
        output_file = f"./{path}/TEST_OLS_Regression_Feature_Analysis_{measure.replace(' ', '_')}.xlsx"
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
            """
            # (2) SHAP (add right after fit)

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_preprocessed)

            shap_importance_df = pd.DataFrame({
                "Feature": X_preprocessed.columns,
                "SHAP_Importance": np.abs(shap_values).mean(axis=0),
            }).sort_values(by="SHAP_Importance", ascending=False)

            shap_output_file = f"./{path}/RF_SHAP_Importance_{measure.replace(' ', '_')}.xlsx"
            shap_importance_df.to_excel(shap_output_file, index=False)
            """
def concat_regression_results(model_based, dataset_based, path, model_name, dataset_name, sensitive_feature):

    # List of uploaded files and their corresponding target measures
    files = {
        "Value Unfairness": "TEST_OLS_Regression_Feature_Analysis_Value_Unfairness_of_sensitive_attribute.xlsx",
        "Overestimation Unfairness": "TEST_OLS_Regression_Feature_Analysis_Overestimation_Unfairness_of_sensitive_attribute.xlsx",
        "Differential Fairness": "TEST_OLS_Regression_Feature_Analysis_Differential_Fairness_of_sensitive_attribute.xlsx",
        "Generalized Cross Entropy": "TEST_OLS_Regression_Feature_Analysis_Generalized_Cross_Entropy.xlsx",
        'KS Statistic': "TEST_OLS_Regression_Feature_Analysis_KS_Statistic_of_sensitive_attribute.xlsx",
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
    if (model_based == True) & (dataset_based == False):
        # Save the concatenated dataframe to a single Excel file
        output_file = f"RQ2_OLS_Regression_Feature_Analysis_fairness_measure_based_{model_name}.xlsx"
    elif (model_based == False) & (dataset_based == True):
        # Save the concatenated dataframe to a single Excel file
        output_file = f"RQ2_OLS_Regression_Feature_Analysis_fairness_measure_based_{dataset_name}_{sensitive_feature}.xlsx"
    elif (model_based == False) & (dataset_based == False):
        # Save the concatenated dataframe to a single Excel file
        output_file = "TEST_RQ1_OLS_Regression_Feature_Analysis_fairness_measure_based.xlsx"

    concatenated_df.to_excel(f"./{path}/"+output_file, index=False)

def run_regression_for_accuracy_measures(data, model_based, path, run_random_forest=False):

    dropped_accuracy_metrics = [
        'hit@5',
        'ndcg@5',
        'recall@5',
        'mrr@5'
    ]
    accuracy_measure= 'hit@5'
    data_temp = data[data["Is Filtered"] == "Yes"]

    if (model_based == True) & (dataset_based == False):
        data_temp = data_temp.drop(
            columns=dropped_accuracy_metrics +
                    ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                     'Subset ID', 'Model Name'])
        categorical_features = ["Sensitive Feature", "Dataset"]
    elif (model_based == False) & (dataset_based == True):
        data_temp = data_temp.drop(
            columns=dropped_accuracy_metrics +
                    ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                     'Subset ID', "Sensitive Feature", "Dataset"])
        categorical_features = ["Model Name"]

    elif (model_based == False) & (dataset_based == False):
        data_temp = data_temp.drop(
            columns=dropped_accuracy_metrics +
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

    # FOR POLYNOMIAL FEATURES
    # Create interaction features
    #poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    #X_poly = poly.fit_transform(X_preprocessed)
    #interaction_feature_names = poly.get_feature_names_out(X_preprocessed.columns)
    #X_poly_df = pd.DataFrame(X_poly, columns=interaction_feature_names)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_preprocessed) # MAKE İT X_poly_df

    # Add a constant for the OLS regression
    X_with_constant = sm.add_constant(X_scaled)
    #features_with_const = ["const"] + list(interaction_feature_names) # UNCOMMENT FOR POLYNOMIAL
    features_with_const = ["const"] + list(X_preprocessed.columns)

    y = data.loc[data_temp.index, accuracy_measure]

    ols_model = sm.OLS(y, X_with_constant).fit()

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
    output_file = f"./{path}/OLS_Regression_Feature_Analysis_{accuracy_measure.replace(' ', '_')}.xlsx"
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

        rf_output_file = f"./{path}/RF_Feature_Importance_{accuracy_measure.replace(' ', '_')}.xlsx"
        rf_importance_df.to_excel(rf_output_file, index=False)

def concat_accuracy_based_regression_results(model_based ,model_name, path):
    # List of uploaded files and their corresponding target measures
    files = {
        "hit@5": "OLS_Regression_Feature_Analysis_hit@5.xlsx"
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
    if model_based == True:
        # Save the concatenated dataframe to a single Excel file
        output_file = f"RQ4_OLS_Regression_Feature_Analysis_accuracy_metric_based_{model_name}.xlsx"
    elif model_based == False:
        # Save the concatenated dataframe to a single Excel file
        output_file = f"RQ4_OLS_Regression_Feature_Analysis_accuracy_metric_based.xlsx"

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

if (model_based == True) & (dataset_based == False):

    model_list = ["NFCF", "FOCF", "PFCN_MLP"]
    for i in model_list:
        data = pd.read_csv("df_regression.csv", index_col=0)
        data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)

        data = data[data["Model Name"]==i]
        run_regression_for_fairness_measures(
            data,
            model_based,
            dataset_based,
            "fairness_measure_based_RQ2",
            run_random_forest=run_random_forest,
        )
        concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based_RQ2", model_name=i, dataset_name=None, sensitive_feature=None)
        print("RQ1-Model Based DONE")

        data = pd.read_csv("df_regression.csv", index_col=0)
        data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
        data = data[data["Model Name"] == i]
        run_regression_for_accuracy_measures(data, model_based, "accuracy_metric_based_RQ2", run_random_forest=run_random_forest)
        concat_accuracy_based_regression_results(model_based=model_based, model_name=i, path="accuracy_metric_based_RQ2")
        print("RQ4-Model Based DONE")

elif (model_based == False) & (dataset_based == True):
    
    data = pd.read_csv("df_regression.csv", index_col=0)
    data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
    data = data[(data["Dataset"]=="BX") & (data["Sensitive Feature"]=="Age")]
    run_regression_for_fairness_measures(
        data,
        model_based,
        dataset_based,
        "fairness_measure_based_RQ2",
        run_random_forest=run_random_forest,
    )
    concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based_RQ2",
                              model_name=None, dataset_name="BX", sensitive_feature="Age")
    print("RQ2-BX-Age DONE")
    print("--------------------------------")
    data = pd.read_csv("df_regression.csv", index_col=0)
    data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
    data = data[(data["Dataset"]=="ml1m") & (data["Sensitive Feature"]=="Age")]
    run_regression_for_fairness_measures(
        data,
        model_based,
        dataset_based,
        "fairness_measure_based_RQ2",
        run_random_forest=run_random_forest,
    )
    concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based_RQ2",
                              model_name=None, dataset_name="ML1M", sensitive_feature="Age")
    print("RQ2-ML1M-Age DONE")
    print("--------------------------------")
    data = pd.read_csv("df_regression.csv", index_col=0)
    data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
    data = data[(data["Dataset"]=="ml1m") & (data["Sensitive Feature"]=="Gender")]
    run_regression_for_fairness_measures(
        data,
        model_based,
        dataset_based,
        "fairness_measure_based_RQ2",
        run_random_forest=run_random_forest,
    )
    concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based_RQ2",
                              model_name=None, dataset_name="ML1M", sensitive_feature="Gender")
    print("RQ2-ML1M-Gender DONE")
    print("--------------------------------")

elif (model_based == False) & (dataset_based == False):
    
    data = pd.read_csv("df_regression.csv", index_col=0)
    data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
    run_regression_for_fairness_measures(
        data, model_based, dataset_based, "fairness_measure_based", run_random_forest=False)

    concat_regression_results(model_based=model_based, dataset_based=dataset_based, path="fairness_measure_based", model_name=None, dataset_name=None, sensitive_feature=None)
    print("RQ1 DONE")
    """
    data = pd.read_csv("df_regression.csv", index_col=0)
    data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
    # Research Question 2
    run_regression_for_accuracy_measures(data, model_based, "accuracy_metric_based", run_random_forest=run_random_forest)
    concat_accuracy_based_regression_results(model_based, "", path="accuracy_metric_based")
    print("RQ4 DONE")
    """