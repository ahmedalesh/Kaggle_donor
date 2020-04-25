import argparse
import pandas as pd
import os

from model.model import Model
from model.gru_att_model import GruAttModel
from model.lgbm_model import LGBMModel

#args.parse directory
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Path to output directory')
    parser.add_argument('--t', '-test', action='store_true', help='test')

    args = parser.parse_args()
    print('loading data')
    df = pd.read_csv('data/test.csv', delimiter=',') if args.t else pd.read_csv('data/train.csv', delimiter=',')
    value = 3
    if value == 1:
        model = Model()
        dp = model.DP()
        df_processed = dp.process(df, args.t)
    elif value == 2:
        model = LGBMModel()
        dp = model.DP()
        df_processed = dp.process(df, args.t)
    else:
        model = GruAttModel()
        dp = model.DP()
        df_processed = dp.process(df, args.t)

    if not args.t:
        model.train(df_processed)
    else:
        df = model.test(df_processed)
        sample_submission = pd.read_csv('data/sample_submission.csv')
        if not os.path.isdir(args.dir):
            os.mkdir(args.dir)
        sample_submission['project_is_approved'] = df['project_is_approved']
        sample_submission.to_csv(os.path.join(args.dir, 'result.csv'), index=False)

if __name__ == '__main__':
    main()