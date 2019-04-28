from util.utilFunc import Raw_DF_Reader, time_delta

from deeputil.Matrixfy import MatrixfyTransformer
import pandas as pd


def naive_value(timestamp):
    start = pd.Timestamp("1900-01-01 00:00:00")
    end = pd.Timestamp("1900-01-01 23:59:59")
    return time_delta(timestamp, start) / time_delta(start, end)


def main():
    t = MatrixfyTransformer(100, naive_value)
    r = Raw_DF_Reader()
    train = r.train.iloc[:41]
    test = r.test.iloc[:34]
    t.fit(train, test)
    train_maps = t.transform(train)
    test_maps = t.transform(test)
    print(train_maps)
    print(test_maps)

if __name__ == "__main__":
    main()
