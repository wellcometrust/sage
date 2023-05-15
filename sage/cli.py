import typer

from sagemaker.predictor import Predictor
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.sklearn import SKLearnModel
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from datetime import datetime
from boto3 import client as boto3_client

app = typer.Typer()


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
def delete(
    endpoint_name: str = typer.Argument(
        "huggingface-pytorch-inference-2021-11-30-22-03-59-851", help="Endpoint Name"
    )
):
    predictor = Predictor(endpoint_name)
    predictor.delete_endpoint()
    typer.secho(f"Deleted endpoint: {endpoint_name}", fg=typer.colors.GREEN)


@app.command()
def predict(
    endpoint_name: str = typer.Argument("wellcome-bert-mesh", help="Endpoint Name"),
    text: str = typer.Argument(
        "The patient has a history of hypertension.", help="Text to classify"
    ),
):
    predictor = Predictor(endpoint_name, serializer=JSONSerializer())
    result = predictor.predict({"text": text})
    typer.secho(f"Result: {result}", fg=typer.colors.GREEN)


@app.command()
def list():
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
def deploy(
    image_uri: str = typer.Argument("huggingface", help="Framework"),
    task: str = typer.Argument("text-classification", help="Task"),
    role: str = typer.Argument(
        "arn:aws:iam::989477750762:role/sagemaker-exec-role",
        help="SageMaker Execution Role",
    ),
    model_path: str = typer.Option("Wellcome/WellcomeBertMesh", help="Model path"),
    entry_point: str = typer.Option("", help="Entry point"),
    instance_count: int = typer.Option(1, help="Instance Count"),
    instance_type: str = typer.Option("ml.t2.medium", help="Instance Type"),
    endpoint_name: str = typer.Option("wellcome-bert-mesh", help="Endpoint Name"),
):
    if not endpoint_name:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"{model_path}-{now}"

    hub = {"HF_MODEL_ID": model_path, "HF_TASK": task}

    if image_uri == "transformers":
        huggingface_model = HuggingFaceModel(
            transformers_version="4.26.0",
            pytorch_version="1.13.1",
            py_version="py39",
            env=hub,
            role=role,
        )

        huggingface_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            role=role,
            endpoint_name=endpoint_name,
        )

    elif "amazonaws" in image_uri:
        model = Model(image_uri=image_uri, role=role)

        model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    elif image_uri == "sklearn":
        sklearn_model = SKLearnModel(
            model_data=model_path,
            entry_point="",  # fill in
            role=role,
            framework_version="1.2-1",
            py_version="py3",
        )

        sklearn_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
    elif image_uri == "pytorch":
        pytorch_model = PyTorchModel(
            model_data=model_path,
            entry_point="",  #  fill in
            role=role,
            framework_version="2.0.0",
            py_version="py310",
        )

        pytorch_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    else:
        raise NotImplementedError(f"Image URI {image_uri} not supported")

    typer.secho(f"Deployed to {endpoint_name}", fg=typer.colors.GREEN)
