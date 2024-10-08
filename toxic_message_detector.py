from pydantic import BaseModel, field_validator
from typing import List
from sagemaker_endpoint import SagemakerEndpoint
from pydantic import Field


class ToxicMessageResponse(BaseModel):
    label: str = Field()
    score: float = Field()

    def is_toxic(self):
        return self.score > 0.5


class ToxicMessageDetector:
    def __init__(self):
        sagemaker_endpoint = SagemakerEndpoint()
        self.classifier = sagemaker_endpoint.get_predictor("huggingface-pytorch-inference-2024-07-05-15-23-12-631")

    def __call__(self, text: str):
        print(len(text))
        try:
            result = self.classifier.predict({
                "inputs": {
                    "text": text,
                }
            })
        except Exception as e:
            print(e)
            raise e
        return ToxicMessageResponse(**result)
