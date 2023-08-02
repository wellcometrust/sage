from transformers import AutoModel, AutoTokenizer
from sagemaker.predictor import Predictor


class CustomPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session, serializer=None, deserializer=None,  **kwargs):
        """Initialize a ``Predictor``.

               Behavior for serialization of input data and deserialization of
               result data can be configured through initializer arguments. If not
               specified, a sequence of bytes is expected and the API sends it in the
               request body without modifications. In response, the API returns the
               sequence of bytes from the prediction result without any modifications.

               Args:
                   endpoint_name (str): Name of the Amazon SageMaker endpoint to which
                       requests are sent.
                   sagemaker_session (sagemaker.session.Session): A SageMaker Session
                       object, used for SageMaker interactions (default: None). If not
                       specified, one is created using the default AWS configuration
                       chain.
                   serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                       serializer object, used to encode data for an inference endpoint
                       (default: :class:`~sagemaker.serializers.IdentitySerializer`).
                   deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                       deserializer object, used to decode data from an inference
                       endpoint (default: :class:`~sagemaker.deserializers.BytesDeserializer`).
               """
        super().__init__(endpoint_name, sagemaker_session, serializer, deserializer, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("Wellcome/WellcomeBertMesh")
        self.model = AutoModel.from_pretrained("Wellcome/WellcomeBertMesh", trust_remote_code=True)

    def predict(
        self,
        data,
        initial_args=None,
        target_model=None,
        target_variant=None,
        inference_id=None,
        custom_attributes=None,
    ):
        """Return the inference from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified when creating the
                Predictor, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then sends the bytes in the request body as is.
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint`` call. Default is None (no default
                arguments).
            target_model (str): S3 model artifact path to run an inference request on,
                in case of a multi model endpoint. Does not apply to endpoints hosting
                single model (Default: None)
            target_variant (str): The name of the production variant to run an inference
                request on (Default: None). Note that the ProductionVariant identifies the
                model you want to host and the resources you want to deploy for hosting it.
            inference_id (str): If you provide a value, it is added to the captured data
                when you enable data capture on the endpoint (Default: None).
            custom_attributes (str): Provides additional information about a request for an
                inference submitted to a model hosted at an Amazon SageMaker endpoint.
                The information is an opaque value that is forwarded verbatim. You could use this
                value, for example, to provide an ID that you can use to track a request or to
                provide other metadata that a service endpoint was programmed to process. The value
                must consist of no more than 1024 visible US-ASCII characters.

                The code in your model is responsible for setting or updating any custom attributes
                in the response. If your code does not set this value in the response, an empty
                value is returned. For example, if a custom attribute represents the trace ID, your
                model can prepend the custom attribute with Trace ID: in your post-processing
                function (Default: None).

        Returns:
            object: Inference for the given input. If a deserializer was specified when creating
                the Predictor, the result of the deserializer is
                returned. Otherwise the response returns the sequence of bytes
                as is.
        """

        text = data['text']
        inputs = self.tokenizer(text, padding="max_length")
        preds = self.model(input_ids=[inputs["input_ids"]])

        id2label = self.model.config.id2label

        prediction = [
            {"label": id2label[label_id], "score": p}
            for label_id, p in enumerate(preds[0].tolist()) if p > 0.5
        ]
        return prediction


