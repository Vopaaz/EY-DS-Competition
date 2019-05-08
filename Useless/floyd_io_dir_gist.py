# DFPreparation.py line 106-113


def __get_filepath(self):
    '''
        Map the parameters of the provider to an identifiable filepath.
    '''
    dir_ = r"/floyd/input/Tmp"
    fname = self.set_.upper() + "-" + "-".join(self.features) + \
        ("-pathfilled" if self.path_filled else "") + ".csv"
    return os.path.join(dir_, fname)

# BaseUtil.py line 26-41


def __get_raw_test(self):
    r'''
        Read the raw test data table.
        Its path should be r"EY-DS-Competition\OriginalFile\data_test\data_test.csv"
    '''

    with open(r"/floyd/input/OriginalFile/data_test/data_test.csv", "r", encoding="utf-8") as f:
        self.test = pd.read_csv(f, index_col=0)


def __get_raw_train(self):
    r'''
        Read the raw train data table.
        Its path should be r"EY-DS-Competition\OriginalFile\data_train\data_train.csv"
    '''
    with open(r"/floyd/input/OriginalFile/data_train/data_train.csv", "r", encoding="utf-8") as f:
        self.train = pd.read_csv(f, index_col=0)


# MatrixProvider.py line 57-81
def __get_indexpath(self):
    dir_ = r"/floyd/input/Tmp"
    if self.is_train:
        name = "train_index"
    else:
        name = "test_index"
    if self.fill_path:
        fp = "fill"
    else:
        fp = "nfill"
    fname = name + "-p" + str(self.pixel) + "-" + fp + ".csv"
    return os.path.join(dir_, fname)


def __get_filepath(self):
    dir_ = r"/floyd/input/Tmp"
    if self.is_train:
        name = "train_matrix"
    else:
        name = "test_matrix"
    if self.fill_path:
        fp = "fill"
    else:
        fp = "nfill"
    fname = name + "-p" + str(self.pixel) + "-" + fp + ".npz"
    return os.path.join(dir_, fname)
