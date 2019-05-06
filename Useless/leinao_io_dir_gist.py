# Solution\DFPreparation.py, line 108-115
def __get_filepath(self):
    '''
        Map the parameters of the provider to an identifiable filepath.
    '''
    dir_ = r"/input_dir/datasets/eyds/Tmp"
    fname = self.set_.upper() + "-" + "-".join(self.features) + \
        ("-pathfilled" if self.path_filled else "") + ".csv"
    return os.path.join(dir_, fname)

# Solution\util\utilFunc.py, line 26-41
def __get_raw_test(self):
    r'''
        Read the raw test data table.
        Its path should be r"EY-DS-Competition\OriginalFile\data_test\data_test.csv"
    '''

    with open(r"/input_dir/datasets/eyds/data_test.csv", "r", encoding="utf-8") as f:
        self.test = pd.read_csv(f, index_col=0)

def __get_raw_train(self):
    r'''
        Read the raw train data table.
        Its path should be r"EY-DS-Competition\OriginalFile\data_train\data_train.csv"
    '''
    with open(r"/input_dir/datasets/eyds/data_train.csv", "r", encoding="utf-8") as f:
        self.train = pd.read_csv(f, index_col=0)