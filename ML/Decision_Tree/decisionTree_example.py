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
import lib.decisionTree as DT

# valid values: ["regression", "ID3_binary", "Gini_binary"]
test_dict = dict()
test_dict['algorithm'] = "regression"

if __name__ == "__main__":
    file_dir, test_dir = None, None

    if test_dict['algorithm'] == 'regression':
        file_dir = './dataset/golf_regression.txt'
    elif test_dict['algorithm'] == 'ID3_binary' or test_dict['algorithm'] == 'Gini_binary':
        file_dir = './dataset/golf_binary_classification.txt'
        test_dir = './dataset/record.txt'
    else:
        pass

    df = pd.read_csv(file_dir)
    
    config = {'algorithm':test_dict['algorithm'], 'maxDepth': 3, 'summary': True}
    dTree = DT.DecisionTree(config)
    dTree.handleDecisionTree(df)

    if test_dir:
        record = pd.read_csv(test_dir)
        print( f"Result: {dTree.findDecisions(record)}" )

