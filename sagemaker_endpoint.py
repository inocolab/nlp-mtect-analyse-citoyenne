import boto3

from sagemaker import Predictor
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.base_deserializers import JSONDeserializer
from sagemaker.base_serializers import JSONSerializer


class SagemakerEndpoint:
    def deploy(self, model_id: str, model_task: str, role_name: str):
        iam = boto3.client('iam')
        role = iam.get_role(RoleName=role_name)['Role']['Arn']

        # Hub Model configuration. https://huggingface.co/models
        hub = {
            'HF_MODEL_ID': model_id,
            'HF_TASK': model_task
        }

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
            env=hub,
            role=role,
            transformers_version="4.26",
            pytorch_version="1.13",
            py_version="py39",
        )

        huggingface_model.deploy(
            serverless_inference_config=ServerlessInferenceConfig(6144)
        )

    def get_predictor(self, endpoint_name: str):
        return Predictor(endpoint_name, serializer=JSONSerializer(), deserializer=JSONDeserializer())
