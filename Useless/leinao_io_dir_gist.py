'''
    Archieve for the differences when IO on LeiNao Cloud Computing Platform.
'''


# Solution\Machine\DFPreparation.py, line 108-115
def __get_filepath(self):
    '''
        Map the parameters of the provider to an identifiable filepath.
    '''
    dir_ = r"/input_dir/datasets/eyds/Tmp"
    fname = self.set_.upper() + "-" + "-".join(self.features) + \
        ("-pathfilled" if self.path_filled else "") + ".csv"
    return os.path.join(dir_, fname)

# Solution\util\BaseUtil.py, line 26-41


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

# Solution\util\Submission.py, line 39-56


def save(self, memo=""):
    '''
        Save the result DataFrame to csv file.
        The target diretory is "Result". The file will be named by monthday-hour-minute-second.

        Parameters:
            - memo: A string that describes this result DataFrame, it will be written in the memo.txt under the Result dir.
    '''
    if not os.path.exists(r"/userhome/output_dir"):
        os.mkdir(r"/userhome/output_dir", 777)

    filename = datetime.datetime.now().strftime(r"%m%d-%H-%M-%S") + ".csv"
    filepath = os.path.join("output_dir", filename)
    self.result.to_csv(filepath, encoding="utf-8",
                       index=False, line_terminator="\n")

    with open(os.path.join("output_dir", "memo.txt"), "a+", encoding="utf-8") as f:
        f.write(filename)
        f.write("\t")
        f.write(str(memo))
        f.write("\n")
