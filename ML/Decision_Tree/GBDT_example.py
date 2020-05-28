import pandas as pd
import lib.GBDT as GBDT


if __name__ == "__main__":
    file_path, test_path = None, None

    file_path = './dataset/golf_regression.txt'
    test_path = './dataset/golf_regression_record.txt'

    df = pd.read_csv(file_path)
    gbdt = GBDT.GBDT()
    gbdt.fit(df, LR=1.,iters=5)

    if test_path:
        records = pd.read_csv(test_path)
        rets = gbdt.findDecisions(records)
        print( f"Result: {rets}")




