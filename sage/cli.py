import typer
import requests

from sagemaker.predictor import Predictor
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.sklearn import SKLearnModel
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.local import LocalSession
from datetime import datetime
from boto3 import client as boto3_client

app = typer.Typer()


def _deploy_locally(model):
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}
    model.sagemaker_session = sagemaker_session


@app.command()
def logs(endpoint_name):
    # return logs given log group name
    client = boto3_client("logs")
    response = client.describe_log_streams(
        logGroupName="/aws/sagemaker/Endpoints/" + endpoint_name,
        orderBy="LastEventTime",
        descending=True,
    )

    log_streams = response["logStreams"]
    print(log_streams)
    if len(log_streams) == 0:
        return None
    log_stream_name = log_streams[0]["logStreamName"]
    response = client.get_log_events(
        logGroupName="/aws/sagemaker/Endpoints/" + endpoint_name,
        logStreamName=log_stream_name,
        startFromHead=True,
    )

    for event in response["events"]:
        print(event["message"])


@app.command()
def delete(endpoint_name: str = typer.Argument(help="Endpoint Name")):
    predictor = Predictor(endpoint_name)
    predictor.delete_endpoint()
    typer.secho(f"Deleted endpoint: {endpoint_name}", fg=typer.colors.GREEN)


@app.command()
def predict(
    endpoint_name: str = typer.Argument("wellcome-bert-mesh", help="Endpoint Name"),
    text: str = typer.Argument(
        "The patient has a history of hypertension.", help="Text to classify"
    ),
    local: bool = typer.Option(False, help="Is local"),
    port: int = typer.Option(8080, help="Port")
):
    if local:
        # Do a http request
        req = requests.post(
            f"http://localhost:{port}/invocations",
            json={"text": text},
            headers={"Content-Type": "application/json"},
        )

        result = req.json()
        typer.secho(f"Result: {result}", fg=typer.colors.GREEN)
    else:
        predictor = Predictor(endpoint_name,
                              sagemaker_session=None,
                              serializer=JSONSerializer()
                              )
        try:
            result = predictor.predict(data={"text": text})
            typer.secho(f"Result: {result}", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"The Prediction endpoint model returned an error. If the model is big, this may happen due "
                        f"to the model not being ready yet. Try again after some minutes. If the problem persists, "
                        f"analyse the logs using\n\n`sage logs <endpoint-name>`.", fg=typer.colors.YELLOW)
            typer.secho(f"Error: {e}")


@app.command()
def list_endpoints():
    sagemaker_client = boto3_client("sagemaker")

    response = sagemaker_client.list_endpoints(
        SortBy="CreationTime", SortOrder="Descending"
    )

    for endpoint in response["Endpoints"]:
        typer.secho("-" * 10, fg=typer.colors.GREEN)
        typer.secho(
            f"Endpoint name: {endpoint['EndpointName']}\
                    \nEndpoint status: {endpoint['EndpointStatus']}",
            fg=typer.colors.GREEN,
        )


@app.command()
def list_models():
    sagemaker_client = boto3_client("sagemaker")

    response = sagemaker_client.list_models(
        SortBy="CreationTime", SortOrder="Descending"
    )

    for model in response["Models"]:
        typer.secho("-" * 10, fg=typer.colors.GREEN)
        typer.secho(
            f"Model name: {model['ModelName']}",
            fg=typer.colors.GREEN,
        )


@app.command()
def list_tags(resource_arn: str = typer.Argument("", help="Resource arn")):
    if resource_arn.strip() == '':
        typer.secho("Please specify a resource arn to get tags from.", fg=typer.colors.RED)
        exit(-1)
    sagemaker_client = boto3_client("sagemaker")

    typer.secho(sagemaker_client.list_tags(ResourceArn=resource_arn), fg=typer.colors.GREEN)


def _get_tags_from_endpoint(endpoint: str = typer.Argument("", help="Endpoint name")):
    if endpoint.strip() == '':
        typer.secho("Please specify an Endpoint name to get tags from.", fg=typer.colors.RED)
        exit(-1)

    sagemaker_client = boto3_client("sagemaker")
    description = sagemaker_client.describe_endpoint(EndpointName=endpoint)
    return sagemaker_client.list_tags(ResourceArn=description['EndpointArn'])


@app.command()
def list_tags_from_endpoint(endpoint: str = typer.Argument("", help="Endpoint name")):
    tags = _get_tags_from_endpoint(endpoint)
    typer.secho(tags, fg=typer.colors.GREEN)


@app.command()
def describe_endpoint(endpoint_name: str = typer.Argument("", help="Endpoint name")):
    sagemaker_client = boto3_client("sagemaker")
    typer.secho(sagemaker_client.describe_endpoint(EndpointName=endpoint_name), fg=typer.colors.GREEN)


@app.command()
def deploy(
    image_uri: str = typer.Argument("huggingface", help="Framework"),
    task: str = typer.Argument("text-classification", help="Task"),
    role: str = typer.Argument(help="SageMaker Execution Role"),
    model_path: str = typer.Option("Wellcome/WellcomeBertMesh", help="Model path"),
    entry_point: str = typer.Option("", help="Entry point"),
    instance_count: int = typer.Option(1, help="Instance Count"),
    instance_type: str = typer.Option("ml.t2.medium", help="Instance Type"),
    endpoint_name: str = typer.Option("wellcome-bert-mesh", help="Endpoint Name")
):
    if not endpoint_name:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"{model_path}-{now}"

    if image_uri.lower().strip() in ["transformers", "huggingface"]:

        env = {"HF_TASK": task}
        model_data = model_path

        if not model_path.startswith('s3:'):
            typer.secho("Your model is not stored in s3. The inference from models from Hugging Face hub is not"
                        "completely supported by Sagemaker. If it fails, please see README.md section 'Uploading"
                        " custom Hugging Face model to S3' to solve this problem.")

        if not model_path.startswith('s3:'):
            env = {"HF_MODEL_ID": model_path, "HF_TASK": task}
            model_data = None

        huggingface_model = HuggingFaceModel(
            transformers_version="4.26.0",
            pytorch_version="1.13.1",
            py_version="py39",
            entry_point=entry_point,
            model_data=model_data,
            env=env,
            role=role
        )

        if instance_type.lower().strip() == 'local':
            _deploy_locally(huggingface_model)

        huggingface_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            role=role,
            endpoint_name=endpoint_name,
        )

    elif image_uri.lower().strip() in ["amazonaws", "amazon", "aws"]:
        model = Model(image_uri=image_uri,
                      role=role
                      )

        if instance_type.lower().strip() == 'local':
            _deploy_locally(model)

        model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    elif image_uri.lower().strip() == "sklearn":
        sklearn_model = SKLearnModel(
            model_data=model_path,
            entry_point=entry_point,
            role=role,
            framework_version="1.2-1",
            py_version="py3"
        )

        if instance_type.lower().strip() == 'local':
            _deploy_locally(sklearn_model)

        sklearn_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    elif image_uri.lower().strip() == "pytorch":
        pytorch_model = PyTorchModel(
            model_data=model_path,
            entry_point=entry_point,
            role=role,
            framework_version="2.0.0",
            py_version="py310"
        )

        if instance_type.lower().strip() == 'local':
            _deploy_locally(pytorch_model)

        pytorch_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    else:
        raise NotImplementedError(f"Image URI {image_uri} not supported.\n"
                                  f"Supported: pytorch | transformers | aws | sklearn")

    typer.secho(f"Deployed to {endpoint_name}", fg=typer.colors.GREEN)
