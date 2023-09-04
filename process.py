import random
import pandas as pd

def shuffle():
    df1 = pd.read_csv('data/angle_variance_real.csv')
    df1['label'] = 'real'
    df2 = pd.read_csv('data/angle_variance_spoof.csv')
    df2['label'] = 'spoof'

    combined_df = pd.concat([df1, df2], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42)
    shuffled_df.to_csv('data/combined_and_shuffled.csv', index=False)
    print("file saved")

if __name__ == "__main__":
    shuffle()