class treeLevel:
    def __init__(self):
        self.level = None
        self.isDone = None
        self.doneValue = None                # for regression value
        self.node_name = None                   # ["Sunny"] or ["Humidity"]
        self.pre_node_name = None
        self.branches = dict()               # {"Sunny": ["undone", "Humidity"], "Overcast":["done","Yes"], "Rain":["undone","Rain"]}
        self.feature_samples = dict()        # {"Sunny": xxx, "Overcast": xxx, "Rain": xxx} or {"Wind":xxx}

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