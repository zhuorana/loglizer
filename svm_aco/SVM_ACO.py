import sys
import time

sys.path.append('../')
from loglizer.models import SVM
from loglizer import dataloader, preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

C_VALUE_RANGE_STARTING_INDEX = -5
GAMMA_VALUE_RANGE_STARTING_INDEX = -5


def get_c_gamma(_x, _y):
    return \
        pow(2, C_VALUE_RANGE_STARTING_INDEX + _x), \
        pow(2, GAMMA_VALUE_RANGE_STARTING_INDEX + _y)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session',
                                                                train_ratio=0.9,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    tic = time.time()
    best_precision = 0

    for i in range(0, 10):
        for j in range(0, 10):
            c, gamma = get_c_gamma(i, j)
            model = SVM(C=c, gamma=gamma)
            model.fit(x_train, y_train)
            precision, recall, f1 = model.evaluate(x_test, y_test)
            if precision > best_precision:
                best_precision = precision
    toc = time.time()
    print(f"Spent {toc - tic:0.4f} seconds")

    print(best_precision)
