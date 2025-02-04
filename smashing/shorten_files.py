import pandas as pd

def halve_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    new_df = pd.DataFrame(columns=['lang', 'source_entity', 'target'])
    index = 1
    for index, row in df.iterrows():
        if index % 2 == 0:
            new_df.loc[index] = [row['lang'], row['source_entity'], row['target']]

    new_df.to_csv(output_csv, index=False)


halve_csv('smashed_train.csv', 'smashed_train_short.csv')
halve_csv('smashed_val.csv', 'smashed_val_short.csv')
