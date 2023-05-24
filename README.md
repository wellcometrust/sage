# Using the CLI tool

## Quickstart

### 0. Prerequisites

You will need:
- a trained model
- an entrypoint script
- an S3 bucket to store the model

You can create an S3 bucket via:
```bash
aws s3 mb s3://<bucket-name>
```

Next, zip your local model:
```bash
tar -czvf model.tar.gz <path-to-model>
```

Upload the zipped model to S3:
```bash
aws s3 cp model.tar.gz s3://<bucket-name>
```

Create an entrypoint script.
Note that the entrypoint script must implement the:
- `input_fn`: responsible for pre-processing the input
- `model_fn`: responsible for loading the model
- `predict_fn`: responsible for running the model on the input data
- `output_fn`: post-procesing of the model's output

See the example entrypoints in `example_entrypoints` folder for some toy examples.
Note that in `model_fn`, you should refer to the model's path without the `tar.gz` extension, i.e. as if the model were local and already unzipped (SageMaker will automatically unzip the model for you).

### 1. Install dependencies and activate environment

```bash
poetry install
poetry shell
```

### 2. Deploy an infernece endpoint
Currently, we support four frameworks for deployment, namely:
- `aws`: This is where you specify an image URI that is available in ECR. This image will be used as the entrypoint for the deployment app.
- `pytorch`: This is where you specify a path to a PyTorch model in an S3 bucket. Model must be tar-gzip compressed.
- `sklearn`: This is where you specify a path to a scikit-learn model in an S3 bucket. Model must be tar-gzip compressed.
- `huggingface`: This is where a model from the [HuggingFace model repository](https://huggingface.co/models) is deployed. The model name must be specified.

In addition to these supported frameworks, the deployment can be either local or remote. For local deployment, the `--instance-type local` flag must be specified. For remote deployment, specify a SageMaker instance type, e.g. `--instance-type ml.m5.large`.

Example 1: Deploy a PyTorch model locally
```bash
sage deploy pytorch \
    text-classification \
    <arn-role> \
    --model-path <path-of-model-in-s3> \
    --endpoint-name test \
    --instance-type local \
    --entry-point sage/example_entrypoints/pytorch_dummy_entrypoint.py
```

Example 2: Deploy an sklearn model locally
```bash
sage deploy \
    sklearn \
    --model-path <path-to-model-in-s3> \
    --endpoint-name test \
    --instance-type local
```

Example 3: Deploy a custom image from ECR to a remote endpoint
```bash
sage deploy \
    aws \
    <image-uri> \
    --endpoint-name test \
    --instance-type <type-of-instance>
```

### 3. List endpoint

```bash
sage list
```

Initially, the endpoint will be in the `Creating` state. Wait until it is in the `InService` state.

### 4. Run inference

```bash
sage predict --endpoint-name test --text "This is a test sentence."
```

The output of the inference call will be displayed to stdout.


## Commands

### Deploy
Deploys a model to an inference endpoint. The model can be either from a local path or an ECR URI.


```bash
sage deploy \
    <model-path> \
    <image-uri> \
    --endpoint-name <endpoint-name> \
    --instance-type <instance-type> \
    --instance-count <instance-count> \
    --entry-point <entry-point>
```

### List endpoints
Lists all available endpoints, along with their status.
```bash
sage list
```

### Delete endpoint
Deletes a deployed endpoint.
```bash
sage delete \
    --endpoint-name <endpoint-name>
```

### Run inference
Runs inference on a deployed endpoint. Returns the output of the inference call to stdout.

```bash
sage predict \
    --endpoint-name <endpoint-name> \
    --text <text>
```

### Logs
Displays the logs of a deployed endpoint.
```bash
sage logs \
    --endpoint-name <endpoint-name>
```
