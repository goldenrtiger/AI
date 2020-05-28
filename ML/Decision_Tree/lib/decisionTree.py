import lib.treeLevel as TL
import numpy as np
import pandas as pd

class DecisionTree:
    '''
        configure: 
            algorithm: ['ID3_binary', 'Gini_binary', 'regression']
            maxDepth: int
            summary: True or False
    '''
    def __init__(self, configure):
        self.valid_algorithms = ['ID3_binary', 'Gini_binary', 'regression']
        
        self.trees = []
        self.initConfig(configure)

    def initConfig(self, configure):
        algorithm = 'ID3_binary'
        maxDepth = 3
        summary = False
        for key, value in configure.items():
            if key == 'algorithm':
                algorithm = value
                if algorithm not in self.valid_algorithms:            
                    raise ValueError('Invalid algorithm passed. You passed ', algorithm," but valid algorithms are ",valid_algorithms)
            elif key == 'maxDepth':
                maxDepth = value
            elif key == 'summary':
                summary = True
            else:
                pass
        
        self.configure = dict()
        self.configure['algorithm'] = algorithm
        self.configure['maxDepth'] = maxDepth
        self.configure['summary'] = summary

    def summary(self):
        i = 0
        for l in self.trees:
            for tree in l:
                print(f" level: {i}, pre_node_name: {tree.pre_node_name}, root_node_name: {tree.node_name} \n nodes_detail: {tree.branches} \n")
            i += 1

    def createTreeLevelBinary(self, df):
        rows, cols = df.shape[0], df.shape[1]
        features = df.columns
        gains = []
        tree = TL.treeLevel()

        if (df['Target'] == 'Yes').all() or (df['Target'] == 'No').all():
            tree.isDone = True
            return tree

        for feature in features:
            if feature == 'Target':
                continue

            sub_feature_dict_yes = dict()
            sub_feature_dict_no = dict()
            yes_total, no_total = 0, 0            
            sub_feature = df[feature]

            for idx in range(len(sub_feature)):
                str_key = sub_feature[idx]
                if str_key not in sub_feature_dict_yes.keys():
                    sub_feature_dict_yes[str_key] = 0
                if str_key not in sub_feature_dict_no.keys():
                    sub_feature_dict_no[str_key] = 0 
                
                fDict = None
                if df['Target'][idx] == 'Yes':
                    fDict = sub_feature_dict_yes
                    yes_total += 1
                elif df['Target'][idx] == 'No':
                    fDict = sub_feature_dict_no
                    no_total += 1
                
                fDict[str_key] += 1 

            sum = 0     # for gain
            if self.configure['algorithm'] == 'ID3_binary':
                sum = Entropy = -yes_total/rows * np.log2(yes_total/rows) - (no_total/rows) * np.log2(no_total/rows)  
            
            for key, value in sub_feature_dict_yes.items():              
                no_total = 0
                if key in sub_feature_dict_no.keys():
                    no_total = sub_feature_dict_no[key]
                total = value + no_total
   
                if self.configure['algorithm'] == 'Gini_binary':
                    gini = 1- (value/total) ** 2 - (no_total/total) ** 2
                    sum += (total/rows) * gini
                elif self.configure['algorithm'] == 'ID3_binary':
                    Entropy_attribute = 0
                    if value != 0 and no_total != 0: 
                        Entropy_attribute = -(value/total) * np.log2(value/total) - (no_total/total) * np.log2(no_total/total)
                    p_attribute = total/rows
                    sum -= Entropy_attribute * p_attribute                          
                else:
                    raise ValueError(f"algorithm:{self.configure['algorithm']} is wrong, please check")   

            gains.append(sum)

        idx = 0        
        if self.configure['algorithm'] == 'Gini_binary':
            idx = np.argmin(np.array(gains))
        elif self.configure['algorithm'] == 'ID3_binary':
            idx = np.argmax(np.array(gains))
        else:
            raise ValueError(f"algorithm:{self.configure['algorithm']} is wrong, please check")

        tree.node_name = features[idx]      
        item_dict = dict()

        for item in df[tree.node_name]:
            if item not in item_dict.keys():
                item_dict[item] = TL.treeLevel.initBranchValue()

        tree.branches = item_dict

        return tree
        
    def createTreeLevelRegression(self, df):
        rows, cols = df.shape[0], df.shape[1]
        features = df.columns
        sdrs = []
        tree = TL.treeLevel()

        target = df['Target'].tolist()
        sd = np.std(np.array(target))
        average = np.average(np.array(target))
        cv = sd/average

        if abs(cv) < 0.1 or len(target) <= 3 or cols < 3:
            tree.isDone = True
            tree.doneValue = average
            return tree
        
        for feature in features:
            if feature == 'Target':
                continue
            
            sub_feature_dict = dict()
            sub_feature = df[feature]
            for idx in range(len(sub_feature)):
                str_key = sub_feature[idx]
                if str_key not in sub_feature_dict.keys():
                    sub_feature_dict[str_key] = [] # store count of sub_feature values

                sub_feature_dict[str_key].append(df['Target'][idx])
            
            sdr = sd
            for key, value in sub_feature_dict.items():
                size = len(value)
                sub_sd = np.std(np.array(value))
                sdr -= (size/rows) * sub_sd

            sdrs.append(sdr)
        idx = np.argmax(np.array(sdrs))

        tree.node_name = features[idx]
        item_dict = dict()
        for item in df[tree.node_name]:
            if item not in item_dict.keys():
                item_dict[item] = TL.treeLevel.initBranchValue()
        tree.branches = item_dict

        return tree

    def constructDecisionTree(self, df):
        raw_df = df.copy()
        rows, cols = raw_df.shape[0], raw_df.shape[1]
        trees_rets = []
        tree = None

        if self.configure['algorithm'] == 'ID3_binary' or self.configure['algorithm'] == 'Gini_binary':
            createTreeLevel = self.createTreeLevelBinary
        elif self.configure['algorithm'] == 'regression':
            createTreeLevel = self.createTreeLevelRegression
        else:
            pass

        for idx in range(self.configure['maxDepth']):
            if idx == 0:    # Root
                tree = createTreeLevel(raw_df)        
                tree.level = idx
                tree.feature_samples[tree.node_name] = raw_df
                trees_rets.append([tree])
            else:
                pre_trees = trees_rets[-1]
                trees = []
                nodes_name = []
                # when to done
                for pre_tree in pre_trees:
                    if pre_tree.isDone:
                        continue

                    # for branches in pre_tree.feature_condition:
                    branches = pre_tree.branches
                    feature_samples = pre_tree.feature_samples[pre_tree.node_name].copy()
                    for branch_name in branches.keys():       
                        # select branch_name samples
                        samples = feature_samples.values[feature_samples[pre_tree.node_name] == branch_name]                                         
                        columns = feature_samples.columns                 
                        samples = pd.DataFrame(samples, columns=columns)
                        # delete branch_name column
                        samples = samples.drop(pre_tree.node_name, axis=1)
                        tree = createTreeLevel(samples)
                        tree.pre_node_name = branch_name

                        if tree.isDone:
                            if self.configure['algorithm'] == 'ID3_binary' or self.configure['algorithm'] == 'Gini_binary':
                                TL.treeLevel.setBranchDone(branches[branch_name])                                           # update pre_tree feature_condition value
                                TL.treeLevel.setBranchValue(branches[branch_name], samples['Target'][0])   
                            elif self.configure['algorithm'] == 'regression':
                                TL.treeLevel.setBranchDone(branches[branch_name])                                           # update pre_tree feature_condition value
                                TL.treeLevel.setBranchValue(branches[branch_name], tree.doneValue)                                  
                        else:
                            TL.treeLevel.setBranchValue(branches[branch_name], tree.node_name)
                            # tree.feature_samples[tree.feature_name] = feature_samples   # store samples
                            tree.feature_samples[tree.node_name] = samples
                            tree.level = idx
                            trees.append(tree)
                            nodes_name.append(tree.node_name)

                # remove used feature_name in feature_samples

                for tree in trees:
                    for key in tree.feature_samples.keys():
                        sub_feature_name = nodes_name.copy()
                        while tree.node_name in sub_feature_name: sub_feature_name.remove(tree.node_name)
                        tree.feature_samples[key] = tree.feature_samples[key].drop(sub_feature_name, axis=1)

                if trees: trees_rets.append(trees)
     
        return trees_rets 

    def fit(self, df):
        trees = []
        if self.configure['algorithm'] == 'ID3_binary':
            trees = self.constructDecisionTree(df)
        elif self.configure['algorithm'] == 'Gini_binary':
            trees = self.constructDecisionTree(df)
        elif self.configure['algorithm'] == 'regression':
            trees = self.constructDecisionTree(df)
        else:
            raise ValueError(f"algorithm:{self.configure['algorithm']} is wrong, please check")

        self.trees = trees      

        if self.configure['summary']:  
            self.summary()      

    def findDecisions(self, records):
        decisions_rets = []
        value, pre_node_name = None, None
        for record_offset in range(len(records.values)):
            for idx in range(len(self.trees)):
                l = self.trees[idx]
                if idx == 0:        # root
                    for tree in l:
                        if not tree.node_name in records.columns:
                            raise ValueError(f"{tree.node_name} is not in record.columns {record.columns}")
                        feature = records[tree.node_name][record_offset]
                        if not feature in tree.branches.keys():
                            raise ValueError(f"record {feature} is not in {tree.branches}")
                        value = tree.branches[feature]
                        pre_node_name = feature
                            
                else:
                    i = 0
                    for tree in l:
                        i += 1
                        if tree.node_name != TL.treeLevel.getBranchValue(value) or tree.pre_node_name != pre_node_name:
                            continue
                        feature = records[tree.node_name][record_offset]
                        if not feature in tree.branches.keys():
                            raise ValueError(f"record {feature} is not in {tree.branches}")
                        value = tree.branches[feature]
                        pre_node_name = tree.node_name
                        break

                    if i == len(l) + 1:
                        raise ValueError("no match record, please check!")

                if TL.treeLevel.isBranchDone(value):
                    decisions_rets.append(TL.treeLevel.getBranchValue(value))
                    break
        if not decisions_rets:
            raise Exception("Can't find a proper decisions, probably increase maxDepth in tree configuration!")

        return decisions_rets