from transformers import AutoModel, AutoTokenizer
import logging
import json


def model_fn(model_dir):
    return AutoModel.from_pretrained(model_dir, trust_remote_code=True)


def input_fn(data, content_type="application/json"):
    return data


def predict_fn(data, model):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    inputs = tokenizer(data, padding="max_length")
    preds = model(input_ids=[inputs["input_ids"]])
    id2label = model.config.id2label
    prediction = [
        {"label": id2label[label_id], "score": p}
        for label_id, p in enumerate(preds[0].tolist()) if p > 0.5
    ]
    logging.info(f"- Done!")
    return prediction


def output_fn(prediction, accept):
    return json.dumps(prediction)
