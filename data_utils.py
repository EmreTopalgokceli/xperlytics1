# import json
# from datetime import datetime
import pandas as pd
# import datetime as dt
import numpy as np
from collections import Counter
from config import (
    LOAN_DATA_PATH,
    LOAN_DELINQUENCIES_PATH,
    CUSTOMER_DATA_PATH,
    CUSTOMER_FINANCIALS_PATH,
    RANDOM_STATE, CV, DROP_COLS, LGD, CATS
)
from utils.time_utils import start_runtime_tracker
from sklearn.model_selection import train_test_split
import numpy_financial as npf


def flag_default(loan_d):
    '''
    Flags defaulted loans based on delinquency duration.

    This function computes the number of months between 'start_date' and 'end_date'.
    If the difference is exactly 4 months, the loan is flagged as defaulted.
    It also creates a 'default_date' set to the third month after 'start_date' for those that are flagged.

    :param loan_d: loan_delinquencies.csv
    :return:
    '''

    loan_d['start_date'] = pd.to_datetime(loan_d['start_date'])
    loan_d['end_date'] = pd.to_datetime(loan_d['end_date'])

    loan_d['month_diff'] = (loan_d['end_date'].dt.year - loan_d['start_date'].dt.year) * 12 + \
                            (loan_d['end_date'].dt.month - loan_d['start_date'].dt.month)

    loan_d['default'] = (loan_d['month_diff'] == 4).astype(int)

    loan_d['default_date'] = pd.NaT
    loan_d.loc[loan_d['default'] == 1, 'default_date'] = loan_d['start_date'] + pd.DateOffset(months=3)

    return loan_d


def compute_irr(requested_amount, installment, loan_term):
    cash_flows = [-requested_amount] + [installment] * loan_term
    irr = npf.irr(cash_flows)
    if irr is not None and np.isfinite(irr):
        return (1 + irr) ** 12 - 1
    else:
        return np.nan


def calculate_expected_costs_with_actual_interest(row, lgd_ratio=0.8):
    try:
        principal = row['requested_amount']
        term = row['loan_term']
        installment = row['installment']

        annual_rate = compute_irr(principal, installment, term)

        if pd.isna(annual_rate) or annual_rate <= 0:
            return pd.Series([np.nan]*5, index=[
                'expected_profit_principal',
                'expected_profit_interest',
                'expected_loss_principal',
                'expected_total_profit',
                'expected_total_lost'
            ])

        monthly_rate = (1 + annual_rate) ** (1/12) - 1
        total_payment = installment * term
        total_interest = total_payment - principal
        monthly_interest = total_interest / term
        monthly_principal = principal / term

        annual_interest = monthly_interest * 12
        annual_principal = monthly_principal * 12

        expected_profit_principal = annual_principal
        expected_profit_interest = annual_interest
        expected_loss_principal = principal * lgd_ratio

        expected_total_profit = expected_profit_principal + expected_profit_interest
        expected_total_lost = expected_loss_principal

        return pd.Series([
            expected_profit_principal,
            expected_profit_interest,
            expected_loss_principal,
            expected_total_profit,
            expected_total_lost
        ], index=[
            'expected_profit_principal',
            'expected_profit_interest',
            'expected_loss_principal',
            'expected_total_profit',
            'expected_total_lost'
        ])
    except:
        return pd.Series([np.nan]*5, index=[
            'expected_profit_principal',
            'expected_profit_interest',
            'expected_loss_principal',
            'expected_total_profit',
            'expected_total_lost'
        ])


def prepare_loan_data(loan, loan_d, cust):

    """
    This function merges the flagged loan_delinquencies.csv (processed using the flag_default function above)
    with customer_data.csv and loan_data.csv.

    It sets the 'default' flag to 0 for customers who have no recorded delinquencies,
    and adds two derived features to the final dataset:
    - 'age_at_loan': customer's age at the time of the loan (in years),
    - 'months_as_customer_at_loan': how long the customer had been with ING when the loan was issued (in months).

    :param loan: loan_data.csv.
    :param loan_d: Return of flag_default function (above).
    :param cust: customer_data.csv.
    """

    loan = loan.sort_values(by='date')

    loan1 = loan.merge(
        loan_d[['loan_id', 'month_diff', 'default', 'default_date']],
        on='loan_id',
        how='left'
    )
    loan1[['month_diff', 'default']] = loan1[['month_diff', 'default']].fillna(0)

    loan1 = loan1.merge(cust, on='cust_id', how='left')

    loan1['date'] = pd.to_datetime(loan1['date'])
    loan1['birth_date'] = pd.to_datetime(loan1['birth_date'])
    loan1['joined_ING_date'] = pd.to_datetime(loan1['joined_ING_date'])

    loan1['age_at_loan'] = (loan1['date'] - loan1['birth_date']).dt.days // 365
    loan1['months_as_customer_at_loan'] = ((loan1['date'] - loan1['joined_ING_date']) / pd.Timedelta(days=30)).astype(int)

    return loan1


def add_missing_features(df, target_columns):
    """
    """
    df = df.copy()

    missing_flags = df.groupby('cust_id')[target_columns].apply(lambda x: x.isna().all()).astype(int)
    missing_flags.columns = [f"{col}_missing_flag" for col in target_columns]

    missing_flags['num_features_missing'] = missing_flags.sum(axis=1)

    def assign_info_group(n):
        if n == 0:
            return 'high_info'
        elif n <= 2:
            return 'medium_info'
        else:
            return 'low_info'

    missing_flags['info_quality_group'] = missing_flags['num_features_missing'].apply(assign_info_group)

    df = df.merge(missing_flags, on='cust_id', how='left')

    return df


def prepare_cust_f_data(loan1, cust_df):

    """
    Filters the data to include only records dated before up to and including the loan date.,
    so that only information available at decision time is considered.

    :param loan1: Return of prepare_loan_data function (above).
    :param cust_df: customer_financials.csv
    """

    loan1['date'] = pd.to_datetime(loan1['date'])
    cust_df['date'] = pd.to_datetime(cust_df['date'])

    target_columns = ['salary', 'current_acc_balance', 'saving_acc_balance', 'credit_card_balance']
    cust_df = add_missing_features(cust_df, target_columns)

    cust_df = cust_df.sort_values(by=['cust_id', 'date'])

    filtered = cust_df.merge(
        loan1[['cust_id', 'date']].rename(columns={'date': 'loan_date'}),
        on='cust_id'
    )
    filtered = filtered[filtered['date'] <= filtered['loan_date']]

    return filtered



def create_time_window_features(filtered, windows=[3, 6]):

    """
    Aggregates selected financial variables from monthly data up to and including the loan date.

    For each customer, this function extracts:
    - The last recorded value before the loan date (even if the value is missing),
    - The std of all available values before or on the loan date
    - The mean and median of the last 3 and 6 months’ values prior to the loan date (NOT ACTIVE!)

    The output will reflect only the information available at the time of the loan decision,
    with cust_id on the Y-axis and time-based features on the X-axis.

    :param filtered: Return of prepare_cust_f_data function (above) — filtered customer financial data including loan_date.
    :param windows: List of time windows (in number of records) to calculate mean and median over.
    """

    value_cols = ['salary', 'current_acc_balance', 'saving_acc_balance', 'credit_card_balance']

    filtered = filtered.sort_values(['cust_id', 'date'])
    filtered['date'] = pd.to_datetime(filtered['date'])
    filtered['loan_date'] = pd.to_datetime(filtered['loan_date'])

    filtered1 = filtered[['cust_id', 'salary_missing_flag', 'current_acc_balance_missing_flag', 'saving_acc_balance_missing_flag', 'credit_card_balance_missing_flag', 'num_features_missing', 'info_quality_group']].drop_duplicates()

    result_list = []

    for cust_id, group in filtered.groupby('cust_id'):
        loan_date = group['loan_date'].iloc[0]
        data_before_loan = group[group['date'] <= loan_date]

        if data_before_loan.empty:
            continue

        latest_row = data_before_loan.iloc[-1:]
        summary = {
            'cust_id': cust_id,
        }

        for col in value_cols:
            summary[f'{col}_last'] = latest_row[col].values[0]

            std_val = data_before_loan[col].std()
            median_val = data_before_loan[col].median()

            summary[f'{col}_std'] = std_val

            # ! warningleri logger'a yazmayi unutma!!

            summary[f'{col}_cv_robust'] = std_val / median_val if pd.notnull(median_val) and median_val != 0 else np.nan

        result_list.append(summary)

    result_df = pd.DataFrame(result_list)

    filtered = result_df.merge(filtered1, on='cust_id', how='left')

    return filtered


def create_financial_features(loan1, fin_features):
    '''
    Creates financial ratio features and flags for credit and liquidity risk analysis.
    Includes installment-to-income, debt burden ratios, and liquidity stress indicators.

    :param loan1:  Return of prepare_loan_data function.
    :param fin_features: Return of create_time_window_features (above).
    '''

    df = loan1.merge(fin_features, on='cust_id', how='left')

    # Oranlar – eksik değerler NaN kalacak
    df['installment_to_salary'] = df['installment'] / df['salary_last']
    df['installment_to_saving'] = df['installment'] / df['saving_acc_balance_last']
    df['installment_to_current_acc'] = df['installment'] / df['current_acc_balance_last']

    df['requested_amount_to_salary'] = df['requested_amount'] / df['salary_last']
    df['requested_amount_to_saving'] = df['requested_amount'] / df['saving_acc_balance_last']
    df['requested_amount_to_current_acc'] = df['requested_amount'] / df['current_acc_balance_last']

    df['total_balance'] = (
        df['saving_acc_balance_last'].fillna(0) +
        df['current_acc_balance_last'].fillna(0)
    )
    df['total_balance_to_salary'] = df['total_balance'] / df['salary_last']
    df['total_balance_to_requested_amount'] = df['total_balance'] / df['requested_amount']

    df['credit_card_to_salary'] = df['credit_card_balance_last'] / df['salary_last']
    df['credit_card_to_saving'] = df['credit_card_balance_last'] / df['saving_acc_balance_last']
    df['credit_card_to_current_acc'] = df['credit_card_balance_last'] / df['current_acc_balance_last']

    df['has_negative_current_acc'] = (df['current_acc_balance_last'] < 0).astype(int)
    df['high_installment_flag'] = (df['installment_to_salary'] > 0.5).astype(int)

    numerator_3 = df['saving_acc_balance_last'] + df['current_acc_balance_last']
    denominator_3 = df['installment'] + df['credit_card_balance_last']
    df['liquidity_ratio'] = np.where(
        numerator_3.notna() & denominator_3.notna() & (denominator_3 != 0),
        numerator_3 / denominator_3,
        np.nan
    )

    return df


def final_touches(df, CATS):
    datetime_cols = df.select_dtypes(include=["datetime"]).columns
    id_cols = [col for col in df.columns if '_id' in col.lower()]
    df = df.drop(columns=datetime_cols)
    df = df.drop(columns=id_cols, errors='ignore')
    for col in CATS:
        df[col] = df[col].astype('category')
    df = df.drop(DROP_COLS, axis=1)

    return df

def compute_outlier_thresholds(X_train, y_train, features):
    outlier_thresholds = {}
    pos_idx = y_train[y_train == 1].index
    for col in features:
        q1 = X_train.loc[pos_idx, col].quantile(0.25)
        q3 = X_train.loc[pos_idx, col].quantile(0.75)
        iqr = q3 - q1
        low, up = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_thresholds[col] = (low, up)
    return outlier_thresholds


def add_outlier_flags(X, y, features, outlier_thresholds, prefix="out_"):
    X_new = X.copy()
    pos_idx = y[y == 1].index
    for col in features:
        low, up = outlier_thresholds[col]
        flag_col = f"{prefix}{col}"
        X_new[flag_col] = 0
        X_new.loc[pos_idx, flag_col] = (
            (X_new.loc[pos_idx, col] < low) | (X_new.loc[pos_idx, col] > up)
        ).astype(int)
    return X_new


def data_preprocessing():

    start_runtime_tracker()

    loan = pd.read_csv(LOAN_DATA_PATH)
    loan_d = pd.read_csv(LOAN_DELINQUENCIES_PATH)
    cust = pd.read_csv(CUSTOMER_DATA_PATH)
    cust_f = pd.read_csv(CUSTOMER_FINANCIALS_PATH, delimiter=";")
    print("Datasets loaded successfully.")

    loan_d = flag_default(loan_d)
    print('Flagging defaulted loans exceeding a 3-month delinquency.')
    loan1 = prepare_loan_data(loan, loan_d, cust)
    print("Merging loan, delinquency, and customer data.")
    filtered = prepare_cust_f_data(loan1, cust_f)
    print("Filtering customer financial data up to the loan date.")
    fin_features = create_time_window_features(filtered, windows=[3, 6])
    print("Creating time-based financial features.")
    df = create_financial_features(loan1, fin_features)
    print("Creating financial ratio and risk indicator features.")

    df = final_touches(df, CATS)

    X = df.drop(columns=["default"])
    y = df["default"]

    counter = Counter(y)
    neg, pos = counter[0], counter[1]
    scale_pos_weight = neg / pos if pos != 0 else 1

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )

    flag_outliers = [
        'number_accounts',
        'installment_to_current_acc',
        'current_acc_balance_cv_robust',
    ]

    quantiles_dict = {
        'salary_q10': X_train['salary_last'].quantile(0.10),
        'cc_q90': X_train['credit_card_balance_last'].quantile(0.90),
        'saving_q25': X_train['saving_acc_balance_last'].quantile(0.25),
    }

    outlier_thresholds = compute_outlier_thresholds(X_train, y_train, features=flag_outliers)

    X_train = add_outlier_flags(X_train, y_train, flag_outliers, outlier_thresholds)
    X_val = add_outlier_flags(X_val, y_val, flag_outliers, outlier_thresholds)
    X_test = add_outlier_flags(X_test, y_test, flag_outliers, outlier_thresholds)

    train_profit_df = loan1.apply(calculate_expected_costs_with_actual_interest, axis=1).loc[X_train.index].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight, train_profit_df


