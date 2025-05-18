from flask import Flask, render_template, request
import tensorflow as tf # type: ignore
from transformers import AutoTokenizer, TFBertForMaskedLM, BatchEncoding  # type: ignore
import typing

MODEL = "bert-base-uncased"
K = 3

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    
    phrase: str | None = request.form.get('phrase')
    predicted_phrase: str | None = get_prediction(phrase=phrase) if phrase else None

    return render_template('index.html', prediction=predicted_phrase, k=K)

def get_prediction(phrase: str) -> list[str] | None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs: BatchEncoding = tokenizer(phrase, return_tensors="tf")
    mask_token_index : int | None = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        return None
    
    model: TFBertForMaskedLM = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    return [phrase.replace(tokenizer.mask_token, tokenizer.decode([token])) for token in top_tokens]


def get_mask_token_index(mask_token_id: int, inputs: BatchEncoding) -> int | None:
    for index, token in enumerate(inputs["input_ids"][0]):        
        if token == mask_token_id: 
            return index

    return None


if __name__ == "__main__":
    index()
