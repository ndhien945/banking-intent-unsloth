import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, f1_score 

class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        print(f"Loading LoRA checkpoint from: {self.config['model_path']}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config['model_path'],
            max_seq_length=self.config['max_seq_length'],
            load_in_4bit=self.config['load_in_4bit'],
        )
        FastLanguageModel.for_inference(self.model)
        
        self.prompt_template = """You are an AI assistant for a banking customer service. Classify the user's intent into exactly one of the known banking categories.

User Query: {}
Intent: """

    def __call__(self, message):
        inputs = self.tokenizer([self.prompt_template.format(message)], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=64, 
            use_cache=True, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.split("Intent: ")[-1].strip()

def main():
    parser = argparse.ArgumentParser(description="Inference & Evaluation Script")
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--message", type=str, default=None, help="Test a single message. ")
    
    parser.add_argument("--evaluate", type=str, default=None, help="Path to the CSV file for evaluation")
    
    args = parser.parse_args()
    classifier = IntentClassification(model_path=args.config)

    if args.message:
        print("\n" + "="*50)
        print(f"Input   : {args.message}")
        print(f"Predict : {classifier(args.message)}")
        print("="*50 + "\n")

    elif args.evaluate:
        print(f"Calling evaluation on: {args.evaluate}")
        df = pd.read_csv(args.evaluate)
        
        y_true = df['intent_name'].tolist()
        y_pred = []
        for msg in tqdm(df['text'], desc="Predicting"):
            y_pred.append(classifier(msg))
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print("\n" + "="*50)
        print(f"Results for {args.evaluate}")
        print("="*50)
        print(f"Total samples      : {len(df)}")
        print(f"Accuracy         : {acc * 100:.2f}%")
        print(f"F1 Macro Score   : {f1_macro:.5f}")
        print("="*50 + "\n")
    else:
        print("Please provide a message or evaluation file. Example: --evaluate sample_data/test.csv")

if __name__ == "__main__":
    main()