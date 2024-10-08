import os

from langchain.chains.llm import LLMChain
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from typing import List


class CommentSentimentResponse:
    def __init__(self, is_positive: bool):
        self.is_positive = is_positive


class CommentSentimentClassifier:
    def __init__(self, profile: str = "default", model_ids: List[str] = ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", "mistral.mistral-7b-instruct-v0:2"]):
        if os.getenv("AWS_PROFILE") is not None:
            profile = os.getenv("AWS_PROFILE")

        fallback_models = []
        chat = None
        for model_id in model_ids:
            # inference parameters inspired from https://cloud.google.com/vertex-ai/generative-ai/docs/prompt-gallery/samples/classification_sentiment_analysis_15
            bedrock_chat = ChatBedrock(
                credentials_profile_name=profile,
                model_id=model_id,
                model_kwargs={"temperature": 1, "top_p": 0.95, "top_k": 40},
            )
            if not chat:
                chat = bedrock_chat
            else:
                fallback_models.append(bedrock_chat)
        fallback_models.append(chat)
        chat = chat.with_fallbacks(fallback_models * 100)
        self.map_chain = self._build_classification_chain(chat)

    def __call__(self, text: str) -> CommentSentimentResponse:
        result = self.map_chain.invoke(text)["text"]
        is_positive = True
        if "défavorable" in result.lower() or "defavorable" in result.lower():
            is_positive = False
        return CommentSentimentResponse(is_positive)

    def _build_classification_chain(self, llm):
        map_template = """Classifie ce commentaire relatif à un article de loi comme favorable ou défavorable.
        Commentaire: {docs}
        Classification:"""
        map_prompt = PromptTemplate.from_template(map_template)
        return LLMChain(llm=llm, prompt=map_prompt, verbose=True)
