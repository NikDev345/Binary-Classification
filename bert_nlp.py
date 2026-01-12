import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# Load once (important for performance)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)


def tokenize_text(texts, max_len=128):
    """
    Tokenize input texts for BERT
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )


def get_bert_embeddings(texts):
    """
    Returns CLS embeddings from BERT
    """
    bert_model.eval()

    with torch.no_grad():
        inputs = tokenize_text(texts)
        outputs = bert_model(**inputs)

        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
