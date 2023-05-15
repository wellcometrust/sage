from typer.testing import CliRunner
from unittest import mock, TestCase
from sage.cli import app


runner = CliRunner()


class MockBotoSageMakerClient:
    def __init__(self, client_type: str):
        pass

    def list_endpoints(SortBy: str, SortOrder: str):
        return {
            "Endpoints": [
                {
                    "EndpointName": "huggingface-pytorch-inference",
                    "EndpointStatus": "InService",
                },
                {
                    "EndpointName": "sklearn-inference",
                    "EndpointStatus": "InService",
                },
            ]
        }


class MockBotoCloudWatchClient:
    def describe_log_streams(logGroupName: str, orderBy: str, descending: bool):
        return {
            "logStreams": [
                {
                    "logStreamName": "some-logstream-name",
                    "creationTime:": 123456789,
                    "firstEventTimestamp": 123456789,
                    "lastEventTimestamp": 123456789,
                    "lastIngestionTime": 123456789,
                },
                {
                    "logStreamName": "some-logstream-name",
                    "creationTime:": 123456789,
                    "firstEventTimestamp": 123456789,
                    "lastEventTimestamp": 123456789,
                    "lastIngestionTime": 123456789,
                },
            ]
        }

    def get_log_events(logGroupName: str, logStreamName: str, startFromHead: str):
        response = {
            "events": [
                {
                    "timestamp": 123,
                    "message": "this is the first message",
                    "ingestionTime": 123,
                },
                {
                    "timestamp": 124,
                    "message": "this is the second message",
                    "ingestionTime": 124,
                },
            ],
            "nextForwardToken": "sometoken",
            "nextBackwardToken": "anothertoken",
        }

        return response


class MockSagemakerPredictor:
    def __init__(self, endpoint_name: str, serializer=None):
        pass

    def predict(self):
        return {"result": "success"}

    def delete_endpoint():
        return None


class MockModel:
    def __init__(self, endpoint_name: str):
        pass

    def deploy(initial_instance_count, instance_type, endpoint_name, role=None):
        return None


class TestCLICommands(TestCase):
    def test_list(self):
        with mock.patch("sage.cli.boto3_client", return_value=MockBotoSageMakerClient):
            result = runner.invoke(app, ["list"])
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_aws_deployment(self):
        image_uri = (
            "989477750762.dkr.ecr.eu-west-1.amazonaws.com/wellcome-custom-containers"
        )

        with mock.patch("sage.cli.Model", return_value=MockModel):
            result = runner.invoke(
                app,
                [
                    "deploy",
                    image_uri,
                    "--endpoint-name",
                    "test",
                ],
            )
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_hf_deployment(self):
        image_uri = "transformers"
        model_path = "Wellcome/Wellcome-Bert-Mesh"

        with mock.patch("sage.cli.HuggingFaceModel", return_value=MockModel):
            result = runner.invoke(
                app,
                [
                    "deploy",
                    image_uri,
                    "--model-path",
                    model_path,
                    "--endpoint-name",
                    "test",
                ],
            )

        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_unimplemented_deployment(self):
        image_uri = "unimplemented"

        with mock.patch("sage.cli.Model", return_value=MockModel):
            result = runner.invoke(
                app, ["deploy", image_uri, "--endpoint-name", "test"]
            )

        assert isinstance(result.exception, NotImplementedError)

    def test_sklearn_deployment(self):
        image_uri = "sklearn"

        with mock.patch("sage.cli.SKLearnModel", return_value=MockModel):
            result = runner.invoke(
                app,
                [
                    "deploy",
                    image_uri,
                    "--endpoint-name",
                    "test",
                ],
            )
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_pytorch_deployment(self):
        image_uri = "pytorch"

        with mock.patch("sage.cli.PyTorchModel", return_value=MockModel):
            result = runner.invoke(
                app,
                [
                    "deploy",
                    image_uri,
                    "--endpoint-name",
                    "test",
                ],
            )
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_logs(self):
        with mock.patch("sage.cli.boto3_client", return_value=MockBotoCloudWatchClient):
            result = runner.invoke(app, ["logs", "some-endpoint"])
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_predict(self):
        with mock.patch("sage.cli.Predictor", return_value=MockSagemakerPredictor):
            result = runner.invoke(app, ["predict", "nonexistent-endpoint", "text"])
        print(result)
        self.assertEqual(result.exit_code, 0)

    def test_delete(self):
        with mock.patch("sage.cli.Predictor", return_value=MockSagemakerPredictor):
            result = runner.invoke(app, ["delete", "nonexistent-endpoint"])
        print(result)
        self.assertEqual(result.exit_code, 0)
