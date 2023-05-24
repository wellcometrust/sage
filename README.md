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

## Examples
In this example, we will deploy a PyTorch model to a local endpoint.

### 1. Create the model and save it
We provide an example script to load the model and store it in a tar.gz archive. Note that this is a dummy model intended for demonstration only.
You can find the script in `example_entrypoints/save_pt_dummy_model.py`

```bash
python example_entrypoints/save_pt_dummy_model.py
```

Note that you need to run this script from an environment with PyTorch 2.0.0 installed. We purposefully do not include PyTorch in the `pyproject.toml` file as the CLI itself does not need it and we want to keep the dependencies minimal.

### 2. Upload to S3
You should now have a `dummy.pt.tar.gz.` file on your disk. Upload it to S3:

```bash
aws s3 cp dummy.pt.tar.gz s3://<bucket-name>
```

### 3. Deploy the model
Activate your poetry shell and run the deploy command.

```bash
poetry shell

sage deploy \
    pytorch \
    text-classification \
    <arn-role> \
    --model-path s3://<bucket-name>/dummy.pt.tar.gz \
    --endpoint-name test \
    --instance-type local \
    --entry-point sage/example_entrypoints/pytorch_dummy_entrypoint.py
```

This will deploy the dummy pytorch model to your local machine.

### 4. Run inference:

Next we'll run the inference command on this entrypoint.
```bash
sage predict \
    test \
    "This is a test sentence." \
    --local
```
You should get a `Result: 1` output. This is the output of the dummy model.
