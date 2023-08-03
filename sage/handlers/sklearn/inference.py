import argparse
import pickle
import json
import os


def model_fn(model_path: str):
    with open(os.path.join(model_path, "dummy.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def input_fn(input_data, content_type="application/json"):
    data = json.loads(input_data)
    return [data["text"]]


def predict_fn(input_data: dict, model):
    prediction = model.predict([input_data])
    return prediction[0]


def output_fn(prediction, content_type="application/json"):
    # Convert output from np data type to int before returning json
    return json.dumps({"prediction": int(prediction)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args, _ = parser.parse_known_args()
    input = input_fn(args.text)
    model = model_fn(args.model_path)
    output = predict_fn(input, model)
    output = output_fn(output)
    print(output)
