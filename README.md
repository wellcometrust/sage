# Using the CLI tool

## Quickstart

### 0. Prerequisites

You will need to upload to S3 a `.tar.gz` file, including:
- the trained model file(s)
- an inference handler script in `code/`
- an S3 bucket to store the model

#### About the model
It may be just a model (e.g., `pytorch_model.pt`) or a whole set of files as, for example, with Hugging Face models (e.g., `WellcomeBertMesh/...`)
IMPORTANT: If you have a folder, It's very important you don't zip the folder, but the contents of the folder itself.

#### About the handler script.
Note that each model requires an inference handler script, implementing the following Sagemaker interface functions:
- `input_fn`: responsible for pre-processing the input
- `model_fn`: responsible for loading the model
- `predict_fn`: responsible for running the model on the input data
- `output_fn`: post-processing of the model's output

See the example entrypoints in `sage/handlers` folder for examples. Note that in `model_fn`, you should refer to the model's path without the `tar.gz` extension, i.e. as if the model were local and already unzipped (SageMaker will automatically unzip the model for you).

#### Zipping and Uploading to S3
You can create an S3 bucket via:
```bash
aws s3 mb s3://<bucket-name>
```

Go to the folder of your model, and add the `entrypoint` file to a folder inside named `code/`.
Example:
```bash
mkdir <path-to-model>/code
cp sage/handlers/pytorch/inference.py <path-to-model>/code/
```

Next, zip your local model:
```bash
cd <path-to-model>
tar -czvf model.tar.gz *
```

Upload the zipped model to S3:
```bash
aws s3 cp model.tar.gz s3://<bucket-name>
```

### 1. Install dependencies and activate environment

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

### 2. Deploy an inference endpoint
Deploys a model to an inference endpoint. The model can be either from a local path or an ECR URI.


```bash
sage deploy 
    <image-uri> \
    <task> \
    <role> \
    --model-path <model-path> \
    --instance-count <instance-count> \
    --instance-type <instance-type> \
    --endpoint-name <endpoint-name>
```
- `image-uri`: Currently, we support four `image-uri` frameworks for deployment, namely:
   - `aws`: This is where you specify an image URI that is available in ECR. This image will be used as the entrypoint for the deployment app.
   - `pytorch`: This is where you specify a path to a PyTorch model in an S3 bucket. Model must be tar-gzip compressed.
   - `sklearn`: This is where you specify a path to a scikit-learn model in an S3 bucket. Model must be tar-gzip compressed.
   - `huggingface`: This is where a model from the [HuggingFace model repository](https://huggingface.co/models) is deployed. The model name must be specified.
- `task` means the type of task, by default `text-classification`
- `role` here means your `arn:...` role with permissions for sagemaker
- `model-path` is the path to `s3` where your model lies. You can also use model names if using hugging face, 
- but many of them are not supported.
- `instance-type` is the type of instance from AWS (e.g, `ml.m5.4xlarge`), or `local` to run everything locally in a Docker Container
- `endpoint-name` give a name to your endpoint. This will be used when calling prediction.


In addition to these supported frameworks, the deployment can be either local or remote. For local deployment, the `--instance-type local` flag must be specified. For remote deployment, specify a SageMaker instance type, e.g. `--instance-type ml.m5.large`.

### 3. List endpoint

```bash
sage list
```

Initially, the endpoint will be in the `Creating` state. Wait until it is in the `InService` state.

### 4. Run inference

```bash
sage predict <endpoint-name> "This is a test sentence."
```

The output of the inference call will be displayed to stdout.


## Other commands

### Logs
Displays the logs of a deployed endpoint.
```bash
sage logs \
    --endpoint-name <endpoint-name>
```

### Delete endpoint
Deletes a deployed endpoint.
```bash
sage delete \
    <endpoint-name>
```

## Examples
In this section, we will deploy several models as examples.

### 1. Custom `Hugging Face` model
Download, for example, `WellcomeBertMesh`:

```bash
git lfs install
git clone https://huggingface.co/Wellcome/WellcomeBertMesh
```

#### Add the custom handler
```bash
cd WellcomeBertMesh
mkdir code
cp sage/handlers/huggingface/inference.py code/
```

#### Zip everything
```bash
cd WellcomeBertMesh
tar -czvf model.tar.gz *
```

#### Upload to S3
```bash
aws s3 cp model.tar.gz s3://<bucket-name>
```

#### Deploy the model
Activate your poetry shell and run the deploy command.

```bash
poetry shell

sage deploy \
    transformers \
    text-classification \
    <arn-role> \
    --model-path s3://<bucket-name>/model.tar.gz \
    --endpoint-name wellcome \
    --instance-type ml.m5.4xlarge 
```

This will deploy the hugging face model to a remote sagemaker endpoint. 

#### Run inference:

Next we'll run the inference command on this entrypoint.
```bash
sage predict \
    wellcome \
    "This is a test sentence."
```
You should get a `Result: b'[{"label": "Humans", "score": 0.8084890842437744}]'` output.

### 2. Local `pytorch` model
We provide an example script to load the model and store it in a tar.gz archive. Note that this is a dummy model intended for demonstration only.
You can find the script in `scripts/save_pt_dummy_model.py`

```bash
python scripts/save_pt_dummy_model.py
```
That script already includes the `pytorch` inference file inside, so no need to copy it.

Note that you need to run this script from an environment with PyTorch 2.0.0 installed. We purposefully do not include PyTorch in the `pyproject.toml` file as the CLI itself does not need it and we want to keep the dependencies minimal.

#### Upload to S3
You should now have a `dummy.pt.tar.gz.` file on your disk. Upload it to S3:

```bash
aws s3 cp dummy.pt.tar.gz s3://<bucket-name>
```

#### Deploy the model
Activate your poetry shell and run the deploy command.

```bash
poetry shell

sage deploy \
    pytorch \
    text-classification \
    <arn-role> \
    --model-path s3://<bucket-name>/dummy.pt.tar.gz \
    --endpoint-name test \
    --instance-type local
```

This will deploy the dummy pytorch model to your local machine.

#### Run inference:

Next we'll run the inference command on this entrypoint.
```bash
sage predict \
    test \
    "This is a test sentence." \
    --local
```
You should get a `Result: 1` output. This is the output of the dummy model.
