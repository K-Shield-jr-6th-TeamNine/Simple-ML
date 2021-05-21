import pandas as pd
from glob import glob
from sklearn import preprocessing
from keras.models import load_model
from pandas import DataFrame
def normalize(df):
    TAG_MIN = df[df.columns].min()
    TAG_MAX = df[df.columns].max()
    ndf = df.copy()
    for c in df.columns:
        try:
            if TAG_MIN[c] == TAG_MAX[c]:
                ndf[c] = df[c] - TAG_MIN[c]
            else:
                ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
        except:
            continue
    return ndf

if __name__ == "__main__":
    g = 'target.csv'
    df = pd.read_csv(g)
    df.drop('timestamp', axis=1, inplace=True)
    print(df.columns)
    col = ['protocol','tot_fwd_pkts','tot_bwd_pkts','fwd_pkt_len_max','fwd_pkt_len_min','fwd_pkt_len_mean',
           'fwd_pkt_len_std','bwd_pkt_len_max','bwd_pkt_len_min','bwd_pkt_len_mean','bwd_pkt_len_std',
           'flow_byts_s','flow_pkts_s','pkt_len_min','pkt_len_max','pkt_len_mean','pkt_len_std','psh_flag_cnt',
           'ack_flag_cnt','urg_flag_cnt','down_up_ratio','pkt_size_avg','fwd_seg_size_avg','bwd_seg_size_avg',
           'subflow_fwd_pkts','subflow_bwd_pkts','fwd_act_data_pkts','fwd_seg_size_min']

    x = df[col]

    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(x)

    model = load_model('CNN.model')

#    answer = y
    predictions = model.predict(X)

    predict = []

    benign = 0
    mal = 0

    for prediction in predictions:
        if prediction[0] < 0.7:
            predict.append(0)
            benign += 1
        else:
            predict.append(1)
            mal += 1

    pred = {'Label' : predict}
    pred1 = DataFrame(pred)

    df['Label'] = pred1['Label']

    print(df)

    df.to_csv("CNN.csv", mode='w')
