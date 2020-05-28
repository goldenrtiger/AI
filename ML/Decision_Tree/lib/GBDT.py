
import pandas as pd
import lib.decisionTree as DT
import lib.treeLevel as TL


class GBDT:
    def __init__(self):
        self.dTrees = []

    def fit(self, df, LR = 0.1, iters=10):
        raw_df = df.copy()
        targets = raw_df['Target'].copy().astype("float64")

        for iter in range(iters):
            dTree = DT.DecisionTree({'algorithm':'regression', 'maxDepth':5, 'summary': True})
            trees = dTree.fit(raw_df)
            records = raw_df.values
            columns = raw_df.columns
            for idx in range(len(records)):
                record = records[idx]
                # if record[columns=='Target'] == 0: continue
                df_record = pd.DataFrame([record[0:-1]], columns = columns[0:-1])
                ret = dTree.findDecisions(df_record)[0]
                targets[idx] = LR * (targets[idx] - ret)

            raw_df.loc[:, 'Target'] = targets
            print(f"targets: {targets}")
            self.dTrees.append(dTree)

    def findDecisions(self, records):
        size = len(records)
        rets = [0] * size

        for idx in range(size):
            record = records.values[idx]
            record = pd.DataFrame([record], columns=records.columns)
            for dTree in self.dTrees:
                rets[idx] += dTree.findDecisions(record)[0]

        return rets
