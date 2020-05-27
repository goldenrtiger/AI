class treeLevel:
    level = None
    isDone = None
    doneValue = None                # for regression value
    feature_name = None             # ["Outlook"] or ["Humidity"]
    branches = dict()               # {"Sunny": ["undone", "Humidity"], "Overcast":["done","Yes"], "Rain":["undone","Rain"]}
    feature_samples = dict()        # {"Sunny": xxx, "Overcast": xxx, "Rain": xxx}

    @staticmethod
    def isBranchDone(branch_value):
        return branch_value[0] == 'done'

    @staticmethod
    def getBranchValue(branch_value):
        return branch_value[1]

    @staticmethod
    def setBranchDone(branch_value):
        branch_value[0] = 'done'

    @staticmethod
    def setBranchValue(branch_value, value):
        branch_value[1] = value

    @staticmethod
    def initBranchValue():
        return ['undone', None]