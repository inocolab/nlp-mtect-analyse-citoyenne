import random
import time

from pydantic import BaseModel, field_validator
from typing import List
from sagemaker_endpoint import SagemakerEndpoint
from pydantic import Field
from botocore.exceptions import ClientError

class ToxicMessageResponse(BaseModel):
    label: str = Field()
    score: float = Field()

    def is_toxic(self):
        return self.score > 0.5


class ToxicMessageDetector:
    def __init__(self):
        sagemaker_endpoint = SagemakerEndpoint()
        self.classifier = sagemaker_endpoint.get_predictor("huggingface-pytorch-inference-2024-07-05-15-23-12-631")

    def __call__(self, text: str, max_retries=4, tensor_size=512):
        retries = 0
        if len(text) > tensor_size:
            text = text[:tensor_size]

        while retries < max_retries:
            try:
                result = self.classifier.predict({
                    "inputs": {
                        "text": text,
                    }
                })
                # print("----", result)
                return ToxicMessageResponse(**result)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    print(f"Retrying after {wait_time} seconds, text: {text[0:50]}...")
                    retries += 1
                else:
                    raise e
