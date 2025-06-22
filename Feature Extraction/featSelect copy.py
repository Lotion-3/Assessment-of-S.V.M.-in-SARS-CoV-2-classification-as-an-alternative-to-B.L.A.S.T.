import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# SelectFromModel is not directly used here for ranking, but useful for selection

def get_and_save_top_features(
    csv_filepath="train10(11000)(1)(extractedALT2).csv",
    output_csv_filepath="top_90_features.csv",
    top_n_per_method=30
):
    """
    Loads data, ranks features by three methods, selects the top N from each,
    combines unique top features, and saves them to a new CSV file.

    Args:
        csv_filepath (str): Path to the input CSV file.
        output_csv_filepath (str): Path to save the CSV with selected features.
        top_n_per_method (int): Number of top features to select from each method.

    Returns:
        pd.DataFrame or None: DataFrame containing the selected features and target,
                              or None if an error occurs.
    """
    print(f"--- Loading and Preprocessing Data from: {csv_filepath} ---")
    try:
        df_orig = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None

    if df_orig.empty:
        print(f"Error: The file {csv_filepath} is empty.")
        return None

    df = df_orig.copy() # Work on a copy

    # --- Preprocessing ---
    variant_to_float = {
        'Alpha': 0.0, 'Beta': 1.0, 'Gamma': 2.0, 'Delta': 3.0, 'Epsilon': 4.0,
        'Zeta': 5.0, 'Eta': 6.0, 'Iota': 7.0, 'Lambda': 8.0, 'Mu': 9.0, 'Omicron': 10.0
    }
    df['Variant_Encoded'] = df['Variant'].map(variant_to_float)

    original_rows = len(df)
    df.dropna(subset=['Variant_Encoded'], inplace=True)
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows due to unmapped variants or NaNs in 'Variant_Encoded'.")

    if df.empty:
        print("Error: DataFrame became empty after dropping rows with unmapped variants.")
        return None

    potential_drop_cols = ['id', 'Variant', 'Variant_Encoded']
    feature_cols = [col for col in df.columns if col not in potential_drop_cols]
    
    if not feature_cols:
        print("Error: No feature columns found. Check CSV structure and drop_cols.")
        return None

    X = df[feature_cols].copy()
    y = df['Variant_Encoded'].astype(int)
    id_col = df['id'] # Keep 'id' for the output file
    target_col_orig_name = df['Variant'] # Keep original 'Variant' string for output

    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    all_top_feature_names = set()
    ranked_features_summary = {}

    print(f"\nOriginal number of features after preprocessing: {X.shape[1]}")
    if X.shape[1] == 0:
        print("Error: No features available for ranking.")
        return None
    
    actual_top_n = min(top_n_per_method, X.shape[1]) # Ensure we don't ask for more features than available

    # --- Method 1: Mutual Information ---
    print(f"\n--- Ranking features using Mutual Information (selecting top {actual_top_n}) ---")
    try:
        mi_scores = mutual_info_classif(X_scaled_df, y, random_state=42)
        mi_ranked_df = pd.DataFrame({
            'Feature': X.columns,
            'Mutual_Information_Score': mi_scores
        }).sort_values(by='Mutual_Information_Score', ascending=False).reset_index(drop=True)
        top_mi_features = mi_ranked_df['Feature'].head(actual_top_n).tolist()
        all_top_feature_names.update(top_mi_features)
        ranked_features_summary['Mutual_Information_Top_Features'] = top_mi_features
        print(f"Top {len(top_mi_features)} features by MI: {top_mi_features[:5]}...")
    except Exception as e:
        print(f"Error during Mutual Information calculation: {e}")
        ranked_features_summary['Mutual_Information_Top_Features'] = []

    # --- Method 2: RandomForest Feature Importance ---
    print(f"\n--- Ranking features using RandomForest Importance (selecting top {actual_top_n}) ---")
    try:
        rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        rf_estimator.fit(X_scaled_df, y)
        rf_importances = rf_estimator.feature_importances_
        rf_ranked_df = pd.DataFrame({
            'Feature': X.columns,
            'RandomForest_Importance': rf_importances
        }).sort_values(by='RandomForest_Importance', ascending=False).reset_index(drop=True)
        top_rf_features = rf_ranked_df['Feature'].head(actual_top_n).tolist()
        all_top_feature_names.update(top_rf_features)
        ranked_features_summary['RandomForest_Top_Features'] = top_rf_features
        print(f"Top {len(top_rf_features)} features by RF: {top_rf_features[:5]}...")
    except Exception as e:
        print(f"Error during RandomForest Importance calculation: {e}")
        ranked_features_summary['RandomForest_Top_Features'] = []

    # --- Method 3: L1 Regularization (Lasso) Coefficients ---
    print(f"\n--- Ranking features using L1 (Lasso) Coefficients (selecting top {actual_top_n}) ---")
    try:
        l1_logreg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=300, random_state=42, class_weight='balanced')
        l1_logreg.fit(X_scaled_df, y)
        if l1_logreg.coef_.ndim > 1:
            l1_coeffs = np.sum(np.abs(l1_logreg.coef_), axis=0)
        else:
            l1_coeffs = np.abs(l1_logreg.coef_[0])
            
        l1_ranked_df = pd.DataFrame({
            'Feature': X.columns,
            'L1_Absolute_Coefficient_Sum': l1_coeffs
        }).sort_values(by='L1_Absolute_Coefficient_Sum', ascending=False).reset_index(drop=True)
        l1_ranked_df_nonzero = l1_ranked_df[l1_ranked_df['L1_Absolute_Coefficient_Sum'] > 1e-5] # Consider only non-zero
        
        # Ensure we take top_n from non-zero features, or all non-zero if less than top_n
        top_l1_features = l1_ranked_df_nonzero['Feature'].head(min(actual_top_n, len(l1_ranked_df_nonzero))).tolist()
        all_top_feature_names.update(top_l1_features)
        ranked_features_summary['L1_Top_Features'] = top_l1_features
        print(f"Top {len(top_l1_features)} features by L1 (non-zero coefs): {top_l1_features[:5]}...")
    except Exception as e:
        print(f"Error during L1 Coefficients calculation: {e}")
        ranked_features_summary['L1_Top_Features'] = []

    # --- Combine and Create Output DataFrame ---
    final_selected_feature_list = sorted(list(all_top_feature_names)) # Sort for consistent column order
    
    if not final_selected_feature_list:
        print("No features were selected by any method. Cannot create output file.")
        return None

    print(f"\nTotal unique features selected from top of all methods: {len(final_selected_feature_list)}")
    print(f"These features are (first 10): {final_selected_feature_list[:10]}...")

    # Create the new DataFrame with 'id', selected features, and original 'Variant' string
    # We need to use the original DataFrame (df_orig) to select these columns
    # based on the indices of the rows that were kept after dropping NaNs in Variant_Encoded
    # The df DataFrame has the correct indices.
    
    output_columns = ['id'] + final_selected_feature_list + ['Variant']
    
    # Select rows from df_orig that correspond to the cleaned df's indices
    df_output = df_orig.loc[df.index, output_columns].copy() # Use .loc with original indices

    print(f"\nShape of the output DataFrame: {df_output.shape}")

    try:
        df_output.to_csv(output_csv_filepath, index=False)
        print(f"Successfully saved selected features to: {output_csv_filepath}")
    except Exception as e:
        print(f"Error saving output CSV to {output_csv_filepath}: {e}")
        return None
        
    return df_output, ranked_features_summary

# --- How to use the function ---
if __name__ == "__main__":
    input_file = "train10(11000)(1)(extractedALT2).csv" # Make sure this file exists
    output_file = "top_features_combined.csv"          # Name for the new CSV
    num_features_per_method = 30                       # Top N from each method

    result_df, summary = get_and_save_top_features(
        csv_filepath=input_file,
        output_csv_filepath=output_file,
        top_n_per_method=num_features_per_method
    )

    if result_df is not None:
        print(f"\n--- Summary of Top 5 Features from Each Method (that contributed to the final set) ---")
        for method, top_list in summary.items():
            print(f"\nMethod: {method}")
            if top_list:
                print(top_list[:5])
            else:
                print("No features selected or error occurred for this method.")
        
        print(f"\nOutput DataFrame head:\n{result_df.head().to_string()}")