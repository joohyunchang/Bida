from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class paraphrase:
    def __init__(self, anno_path, device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality", cache_dir=anno_path)
        self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality", cache_dir=anno_path)
        self.device = device
        self.model.to(device)
        
    def generate(self, context):
        text = "paraphrase: "+context + " </s>"
        self.model.eval()
        with torch.no_grad():
            encoding = self.tokenizer.encode_plus(text, max_length =128, padding='max_length', return_tensors="pt")
            input_ids, attention_mask  = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
            beam_output = self.model.generate(
                input_ids=input_ids,attention_mask=attention_mask,
                max_length=128,
                early_stopping=True,
                num_beams=15,
                num_return_sequences=1
            )
            sent = self.tokenizer.decode(beam_output.squeeze(0), skip_special_tokens=True,clean_up_tokenization_spaces=True).strip('paraphrasedoutput:')
        return sent

if __name__ == "__main__":
    anno_path = "./.debug/"
    paraphrase = paraphrase(anno_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
