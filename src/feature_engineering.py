import pandas as pd

def add_engineered_features(df):
    
    df['avg_bill_amt'] = df[
        ['BILL_AMT1','BILL_AMT2','BILL_AMT3',
         'BILL_AMT4','BILL_AMT5','BILL_AMT6']
    ].mean(axis=1)

    df['credit_utilization'] = df['avg_bill_amt'] / df['LIMIT_BAL']

    df['avg_pay_delay'] = df[
        ['PAY_0','PAY_2','PAY_3',
         'PAY_4','PAY_5','PAY_6']
    ].mean(axis=1)

    return df