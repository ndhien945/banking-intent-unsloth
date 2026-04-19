import os
import argparse
import pandas as pd
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Data preprocessing BANKING77")
    parser.add_argument("--num_train_samples", type=int, default=5390, help="Total train samples")
    parser.add_argument("--num_test_samples", type=int, default=1540, help="Total test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_all", action="store_true", help="Use all data")
    args = parser.parse_args()

    print(f"1. Loading dataset...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    label_names = dataset['train'].features['label'].names

    print(f"2. Sampling data...")
    if args.use_all:
        sampled_train = df_train.copy()
        sampled_test = df_test.copy()
    else:
        train_per_class = args.num_train_samples // 77
        test_per_class = args.num_test_samples // 77

        sampled_train = pd.concat([
            group.sample(n=min(len(group), train_per_class), random_state=args.seed) 
            for _, group in df_train.groupby('label')
        ])
        
        sampled_test = pd.concat([
            group.sample(n=min(len(group), test_per_class), random_state=args.seed) 
            for _, group in df_test.groupby('label')
        ])

    print(f"3. Mapping labels to intent names and shuffling...")
    sampled_train['intent_name'] = sampled_train['label'].apply(lambda x: label_names[x])
    sampled_test['intent_name'] = sampled_test['label'].apply(lambda x: label_names[x])
    
    sampled_train = sampled_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sampled_test = sampled_test.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"4. Saving sampled data...")
    os.makedirs('sample_data', exist_ok=True)
    sampled_train[['text', 'label', 'intent_name']].to_csv('sample_data/train.csv', index=False)
    sampled_test[['text', 'label', 'intent_name']].to_csv('sample_data/test.csv', index=False)
    print(f"Done")
if __name__ == "__main__":
    main()