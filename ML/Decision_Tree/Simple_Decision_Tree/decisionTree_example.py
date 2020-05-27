#   
#   example_classification:
#   Outlook         Temp.           Humidity        Wind        Target
#    Sunny          Hot             High            Weak        No            
#    Sunny          Hot             High            Strong      No
#    Overcast       Hot             High            Weak        Yes
#    Rain           Mild            High            Weak        Yes
#    Rain           Cool            Normal          Weak        Yes
#    Rain           Cool            Normal          Strong      No
#    Overcast       Cool            Normal          Strong      Yes
#    Sunny          Mild            High            Weak        No
#    Sunny          Cool            Normal          Weak        Yes
#    Rain           Mild            Normal          Weak        Yes
#    Sunny          Mild            Normal          Strong      Yes
#    Overcast       Mild            High            Strong      Yes
#    Overcast       Hot             Normal          Weak        Yes
#    Rain           Mild            High            Strong      No


#   Outlook         Temp.           Humidity        Wind        Target
#    Rain            Hot              High          False       25                       
#    Rain            Hot              High          True        30                  
#    Overcast        Hot              High          False       46                           
#    Sunny           Mild             High          False       45                          
#    Sunny           Cool             Normal        False       52                      
#    Sunny           Cool             Normal        True        23                           
#    Overcast        Cool             Normal        True        43                              
#    Rain            Mild             High          False       35                         
#    Rain            Cool             Normal        False       38                           
#    Sunny           Mild             Normal        False       46                            
#    Rain            Mild             Normal        True        48                          
#    Overcast        Mild             High          True        52                            
#    Overcast        Hot              Normal        False       44                          
#    Sunny           Mild             High          True        30    


import pandas as pd
import numpy as np

class treeLevel:
    level = None
    isDone = None
    doneValue = None                # for regression value
    feature_name = None             # ["Outlook"] or ["Humidity"]
    feature_condition = dict()      # {"Sunny": "Humidity", "Overcast":"Yes", "Rain":"Rain"}
    feature_samples = dict()        # {"Sunny": xxx, "Overcast": xxx, "Rain": xxx}

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
        for l in self.trees:
            for tree in l:
                print(f"isDone: {tree.isDone} \n root_node_name: {tree.feature_name} \n nodes_detail:{tree.feature_condition} \n")

    def createTreeLevelBinary(self, df):
        rows, cols = df.shape[0], df.shape[1]
        features = df.columns
        gains = []
        tree = treeLevel()

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

        tree.feature_name = features[idx]      
        item_dict = dict()

        for item in df[tree.feature_name]:
            if item not in item_dict.keys():
                item_dict[item] = None
        tree.feature_condition = item_dict

        return tree
        
    def createTreeLevelRegression(self, df):
        rows, cols = df.shape[0], df.shape[1]
        features = df.columns
        sdrs = []
        tree = treeLevel()

        target = df['Target'].tolist()
        sd = np.std(np.array(target))
        average = np.average(np.array(target))
        cv = sd/average

        if cv < 0.1 or len(target) < 3:
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

        tree.feature_name = features[idx]
        item_dict = dict()
        for item in df[tree.feature_name]:
            if item not in item_dict.keys():
                item_dict[item] = None
        tree.feature_condition = item_dict

        return tree

    def constructDecisionTree(self, df):
        raw_df = df.copy()
        rows, cols = raw_df.shape[0], raw_df.shape[1]
        tree = treeLevel()
        trees_rets = []

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
                tree.feature_samples[tree.feature_name] = raw_df
                trees_rets.append([tree])
            else:
                pre_trees = trees_rets[-1]
                trees = []
                # when to done
                for pre_tree in pre_trees:
                    if pre_tree.isDone:
                        continue

                    # for cond_dict in pre_tree.feature_condition:
                    cond_dict = pre_tree.feature_condition
                    for cond in cond_dict.keys():                    
                        feature_samples = pre_tree.feature_samples[pre_tree.feature_name]
                        samples = feature_samples.values[feature_samples[pre_tree.feature_name] == cond]
                        samples = samples[np.where(samples != cond)].reshape(samples.shape[0], -1)                    
                        columns = feature_samples.columns.drop(pre_tree.feature_name)                        
                        samples = pd.DataFrame(samples, columns=columns)
                        tree = createTreeLevel(samples)

                        if tree.isDone:
                            if self.configure['algorithm'] == 'ID3_binary' or self.configure['algorithm'] == 'Gini_binary':
                                cond_dict[cond] = samples['Target'][0]            # update pre_tree feature_condition value
                            elif self.configure['algorithm'] == 'regression':
                                cond_dict[cond] = tree.doneValue
                        else:
                            cond_dict[cond] = tree.feature_name
                            tree.feature_samples[tree.feature_name] = samples   # store samples
                            tree.level = idx
                        trees.append(tree)

                trees_rets.append(trees)
     
        return trees_rets 


    def handleDecisionTree(self, df):
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

    def findDecisions(self, records, decisions=['Yes', 'No']):
        decisions_rets = []
        value = None
        for record_offset in range(len(records.values)):
            for idx in range(len(self.trees)):
                l = self.trees[idx]
                if idx == 0:        # root
                    for tree in l:
                        if not tree.feature_name in records.columns:
                            raise ValueError(f"{tree.feature_name} is not in record.columns {record.columns}")
                        feature = records[tree.feature_name][record_offset]
                        if not feature in tree.feature_condition.keys():
                            raise ValueError(f"record {feature} is not in {tree.feature_condition}")
                        value = tree.feature_condition[feature]
                else:
                    i = 0
                    for tree in l:
                        i += 1
                        if tree.feature_name != value:
                            continue
                        feature = records[tree.feature_name][record_offset]
                        if not feature in tree.feature_condition.keys():
                            raise ValueError(f"record {feature} is not in {tree.feature_condition}")
                        value = tree.feature_condition[feature]
                        break

                    if i == len(l):
                        raise ValueError("no match record, please check!")

                if value in decisions:
                    decisions_rets.append(value)
                    break

        return decisions_rets

    
# valid values: ["regression", "ID3_binary", "Gini_binary"]
test_dict = dict()
test_dict['algorithm'] = "regression"

if __name__ == "__main__":
    file_dir = None
    test_dir = None

    if test_dict['algorithm'] == 'regression':
        file_dir = './golf_regression.txt'
    elif test_dict['algorithm'] == 'ID3_binary' or test_dict['algorithm'] == 'Gini_binary':
        file_dir = './golf_binary_classification.txt'
        test_dir = './record.txt'
    else:
        pass

    df = pd.read_csv(file_dir)
    
    config = {'algorithm':test_dict['algorithm'], 'maxDepth': 3, 'summary': True}
    dTree = DecisionTree(config)
    dTree.handleDecisionTree(df)

    if test_dir:
        record = pd.read_csv(test_dir)
        print( dTree.findDecisions(record) )





