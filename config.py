import os
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 0
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
LGD = 0.8

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "datasets")
LOAN_DATA_PATH = os.path.join(DATA_DIR, "loan_data.csv")
LOAN_DELINQUENCIES_PATH = os.path.join(DATA_DIR, "loan_deliquencies.csv")
CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, "customer_data.csv")
CUSTOMER_FINANCIALS_PATH = os.path.join(DATA_DIR, "customer_financials.csv")

target_column = 'default'

catboost_params = {
    "iterations": [100, 200, 300],
    "depth": [4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [5, 7, 9],
    "border_count": [32, 64, 128],
    "bagging_temperature": [0, 0.5, 1],
    "random_strength": [1, 5, 10],
    "min_data_in_leaf": [10, 20],
    "rsm": [0.7, 0.8]
}



# BASE + product

CATS=['info_quality_group', 'loan_term', 'loan_reason']


classifiers = [
    (
        'CatBoost',
        CatBoostClassifier(
            verbose=False,
            allow_writing_files=False,
            random_state=RANDOM_STATE,
            cat_features=CATS,
            eval_metric='F1'

        ),
        catboost_params
    )
]


DROP_COLS = ['month_diff', 'number_client_calls_to_ING', 'number_client_calls_from_ING',
             'postal_code', 'gender', 'religion', 'employment',
             'salary_std', 'current_acc_balance_std','credit_card_balance_cv_robust','saving_acc_balance_cv_robust',
             'total_balance',
             'installment_to_saving', 'credit_card_to_salary','requested_amount_to_current_acc', 'credit_card_to_current_acc', "credit_card_to_saving",
             'has_negative_current_acc', 'high_installment_flag', 'saving_acc_balance_missing_flag', 'current_acc_balance_missing_flag']





#################################################################################################################
#################################################################################################################
### Other model setups are left as comments below, in case you'd like to check or experiment with them later. ###
#################################################################################################################
#################################################################################################################


# BASE
#
# CATS=['info_quality_group']
#
#
# classifiers = [
#     (
#         'CatBoost',
#         CatBoostClassifier(
#             verbose=False,
#             allow_writing_files=False,
#             random_state=RANDOM_STATE,
#             cat_features=CATS,
#             eval_metric='F1'
#
#         ),
#         catboost_params
#     )
# ]
#
#
# DROP_COLS = ['month_diff', 'number_client_calls_to_ING', 'number_client_calls_from_ING', 'loan_reason', 'loan_term',
#              'postal_code', 'gender', 'religion', 'employment',
#              'salary_std', 'current_acc_balance_std','credit_card_balance_cv_robust','saving_acc_balance_cv_robust',
#              'total_balance',
#              'installment_to_saving', 'credit_card_to_salary','requested_amount_to_current_acc', 'credit_card_to_current_acc', "credit_card_to_saving",
#              'has_negative_current_acc', 'high_installment_flag', 'saving_acc_balance_missing_flag', 'current_acc_balance_missing_flag']


# BASE + demog

# CATS=['info_quality_group', 'postal_code', 'gender', 'religion', 'employment']
#
#
# classifiers = [
#     (
#         'CatBoost',
#         CatBoostClassifier(
#             verbose=False,
#             allow_writing_files=False,
#             random_state=RANDOM_STATE,
#             cat_features=CATS,
#             eval_metric='F1'
#
#         ),
#         catboost_params
#     )
# ]
#
#
# DROP_COLS = ['month_diff', 'number_client_calls_to_ING', 'number_client_calls_from_ING', 'loan_reason', 'loan_term',
#              'salary_std', 'current_acc_balance_std','credit_card_balance_cv_robust','saving_acc_balance_cv_robust',
#              'total_balance',
#              'installment_to_saving', 'credit_card_to_salary','requested_amount_to_current_acc', 'credit_card_to_current_acc', "credit_card_to_saving",
#              'has_negative_current_acc', 'high_installment_flag', 'saving_acc_balance_missing_flag', 'current_acc_balance_missing_flag']


#  BASE + all
#
# CATS=['info_quality_group', 'postal_code', 'gender', 'religion', 'employment', 'loan_reason', 'loan_term']
#
#
# classifiers = [
#     (
#         'CatBoost',
#         CatBoostClassifier(
#             verbose=False,
#             allow_writing_files=False,
#             random_state=RANDOM_STATE,
#             cat_features=CATS,
#             eval_metric='F1'
#
#         ),
#         catboost_params
#     )
# ]
#
#
# DROP_COLS = ['month_diff', 'number_client_calls_to_ING', 'number_client_calls_from_ING',
#              'salary_std', 'current_acc_balance_std','credit_card_balance_cv_robust','saving_acc_balance_cv_robust',
#              'total_balance',
#              'installment_to_saving', 'credit_card_to_salary','requested_amount_to_current_acc', 'credit_card_to_current_acc', "credit_card_to_saving",
#              'has_negative_current_acc', 'high_installment_flag', 'saving_acc_balance_missing_flag', 'current_acc_balance_missing_flag']



