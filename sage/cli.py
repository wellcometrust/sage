import typer
import requests

from sagemaker import Predictor
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


def _get_predictor(custom_predictor_path):
    if custom_predictor_path is None:
        return None
    else:
        mod = __import__(custom_predictor_path, fromlist=['CustomPredictor'])
        klass = getattr(mod, 'CustomPredictor')
        predictor_class = klass
    return predictor_class


def _add_custom_predictor(endpoint_name, custom_predictor_path):
    sagemaker_client = boto3_client("sagemaker")
    endpoint = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_arn = endpoint['EndpointArn']
    sagemaker_client.add_tags(ResourceArn=endpoint_arn, Tags=[
        {'Key': 'custom_predictor_path',
         'Value': custom_predictor_path}
    ])


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
    else:
        custom_predictor = _get_custom_predictor(endpoint_name)
        print(custom_predictor)
        predictor_class = _get_predictor(custom_predictor)
        if predictor_class is None:
            predictor_class = Predictor

        predictor = predictor_class(endpoint_name,
                                    sagemaker_session=None,
                                    serializer=JSONSerializer()
                                    )
        result = predictor.predict(data={"text": text})

    typer.secho(f"Result: {result}", fg=typer.colors.GREEN)


@app.command()
def list_endpoints():
    sagemaker_client = boto3_client("sagemaker")

    response = sagemaker_client.list_endpoints(
        SortBy="CreationTime", SortOrder="Descending"
    )

    for endpoint in response["Endpoints"]:
        print(endpoint)
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
        print(model)
        typer.secho("-" * 10, fg=typer.colors.GREEN)
        typer.secho(
            f"Model name: {model['ModelName']}",
            fg=typer.colors.GREEN,
        )


@app.command()
def list_tags(resource_arn: str = typer.Argument("", help="Resource arn")):
    if resource_arn.strip() == '':
        print("Please specify a resource arn to get tags from.")
        exit(-1)
    sagemaker_client = boto3_client("sagemaker")

    print(sagemaker_client.list_tags(ResourceArn=resource_arn))


def _get_tags_from_endpoint(endpoint: str = typer.Argument("", help="Endpoint name")):
    if endpoint.strip() == '':
        print("Please specify an Endpoint name to get tags from.")
        exit(-1)

    sagemaker_client = boto3_client("sagemaker")
    description = sagemaker_client.describe_endpoint(EndpointName=endpoint)
    return sagemaker_client.list_tags(ResourceArn=description['EndpointArn'])


@app.command()
def list_tags_from_endpoint(endpoint: str = typer.Argument("", help="Endpoint name")):
    tags = _get_tags_from_endpoint(endpoint)
    print(tags)


@app.command()
def _get_custom_predictor(endpoint: str = typer.Argument("", help="Endpoint name")):
    tags = _get_tags_from_endpoint(endpoint)
    result = list(filter(lambda x: x['Key'] == 'custom_predictor_path', tags['Tags']))
    if len(result) > 0:
        if 'Value' in result[0]:
            return result[0]['Value']
    return None


@app.command()
def describe_endpoint(endpoint_name: str = typer.Argument("", help="Endpoint name")):
    sagemaker_client = boto3_client("sagemaker")
    print(sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
    print(sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name))


@app.command()
def deploy(
    image_uri: str = typer.Argument("huggingface", help="Framework"),
    task: str = typer.Argument("text-classification", help="Task"),
    role: str = typer.Argument(help="SageMaker Execution Role"),
    model_path: str = typer.Option("Wellcome/WellcomeBertMesh", help="Model path"),
    entry_point: str = typer.Option("", help="Entry point"),
    instance_count: int = typer.Option(1, help="Instance Count"),
    instance_type: str = typer.Option("ml.t2.medium", help="Instance Type"),
    endpoint_name: str = typer.Option("wellcome-bert-mesh", help="Endpoint Name"),
    custom_predictor_path: str = typer.Option(None, help="Path to a custom predictor file")
):
    if not endpoint_name:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"{model_path}-{now}"

    if image_uri.lower().strip() in ["transformers", "huggingface"]:

        if instance_type.lower().strip() == 'local':
            # `hub` not supported in `local`mode, you need to use s3
            # https://github.com/aws/sagemaker-python-sdk/issues/2743
            huggingface_model = HuggingFaceModel(
                transformers_version="4.26.0",
                pytorch_version="1.13.1",
                py_version="py39",
                model_data=model_path,
                role=role
            )
            _deploy_locally(huggingface_model)
        else:
            hub = {"HF_MODEL_ID": model_path, "HF_TASK": task}
            huggingface_model = HuggingFaceModel(
                transformers_version="4.26.0",
                pytorch_version="1.13.1",
                py_version="py39",
                entry_point=entry_point,
                env=hub,
                role=role
            )

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
            entry_point=entry_point,  # fill in
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
            entry_point=entry_point,  #  fill in
            role=role,
            framework_version="2.0.0",
            py_version="py310",
            predictor_cls=_get_predictor(custom_predictor_path)
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

    if custom_predictor_path is not None:
        _add_custom_predictor(endpoint_name, custom_predictor_path)

    typer.secho(f"Deployed to {endpoint_name}", fg=typer.colors.GREEN)
