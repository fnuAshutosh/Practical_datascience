"""Utility functions for the retail sales EDA pipeline.

This module extracts the data loading, cleaning and feature engineering
steps from the notebook so they can be reused, tested and executed outside
of the interactive environment.
"""

import os
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Read the retail sales data into a DataFrame, inferring format.

    Supports Excel (.xls, .xlsx) and CSV (.csv) by looking at the file
    extension. The notebook originally used Excel, but CSV is easier for
    testing and avoids an optional dependency.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")
    # coerce date column if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic quality checks and clean up the raw DataFrame.

    Currently this function only validates that Total Amount equals
    Price per Unit * Quantity and drops a temporary computed column.
    Additional rules can be added as required.
    """
    df = df.copy()
    # business logic validation
    df['Computed_Total'] = df['Price per Unit'] * df['Quantity']
    mismatch = df[df['Computed_Total'] != df['Total Amount']]
    if len(mismatch) > 0:
        # in production we might log, raise or store the mismatches
        print(f"⚠️  {len(mismatch)} mismatches between computed and reported total")
    df.drop(columns=['Computed_Total'], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the standard set of engineered features used throughout the notebook.

    This mirrors the operations in Section 2 of the notebook. The resulting
    DataFrame is returned as ``df_feat``.
    """
    df_feat = df.copy()
    df_feat['Year']        = df_feat['Date'].dt.year
    df_feat['Month']       = df_feat['Date'].dt.month
    df_feat['Month_Name']  = df_feat['Date'].dt.strftime('%B')
    df_feat['Quarter']     = df_feat['Date'].dt.quarter
    df_feat['Day_of_Week'] = df_feat['Date'].dt.dayofweek
    df_feat['Day_Name']    = df_feat['Date'].dt.strftime('%A')
    df_feat['Week_of_Year']= df_feat['Date'].dt.isocalendar().week.astype(int)
    df_feat['Is_Weekend']  = df_feat['Day_of_Week'].isin([5, 6]).astype(int)

    age_bins   = [0, 24, 34, 44, 54, 100]
    age_labels = ['18–24 (Gen Z)', '25–34 (Millennial)', '35–44 (Gen X)',
                  '45–54 (Boomer)', '55+ (Senior)']
    df_feat['Age_Group'] = pd.cut(df_feat['Age'], bins=age_bins, labels=age_labels)

    # rolling averages require a separate aggregation so the caller can
    # compute them as needed (not part of df_feat)

    target_enc_map = df_feat.groupby('Product Category')['Total Amount'].mean()
    df_feat['ProductCat_TargetEnc'] = df_feat['Product Category'].map(target_enc_map)

    df_feat['Log_Total_Amount'] = np.log1p(df_feat['Total Amount'])
    df_feat['Log_Price_per_Unit'] = np.log1p(df_feat['Price per Unit'])
    df_feat['Gender_Enc'] = (df_feat['Gender'] == 'Male').astype(int)

    return df_feat
