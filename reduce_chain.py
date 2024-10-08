import os
from typing import List, Any

from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_aws import ChatBedrock
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import TokenTextSplitter


class ReduceChain:
    def __init__(self, sentiment_prompt: str, profile: str = "default", model_id: str = "mistral.mistral-large-2402-v1:0", token_max: int = 32000):
        if os.getenv("AWS_PROFILE") is not None:
            profile = os.getenv("AWS_PROFILE")
        self.llm = self._build_chat_mistral(profile, model_id)
        self.token_max = token_max
        self.reduce_documents_chain = self._build_reduce_documents_chain(self.llm, sentiment_prompt)

    def _build_chat_mistral(self, profile: str, model_id: str):
        chat = ChatBedrock(
            credentials_profile_name=profile,
            model_id=model_id,
            model_kwargs={"temperature": 0.1},
        )

        return chat.with_fallbacks([chat] * 100)

    def _build_reduce_documents_chain(self, llm, sentiment_prompt: str):
        reduce_template = f"""Voici un ensemble de commentaires et résumés de commentaires relatifs à une concertation pour un article de loi, délimité par des triple backticks:
        ```{{docs}}```
        Tu es un agent du ministère de la transition écologique, chargé de produire un rapport pour cette concertation.
        Rédige quelques paragraphes en français listant les arguments {sentiment_prompt} à l'article de loi, sans prendre en compte les commentaires qui ne sont pas {sentiment_prompt} à l'article de loi.
        Réponse:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )
        return ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=self.token_max,
        )

    def _load_dataset(self, data: List[Any]):
        documents: List[Document] = [Document(page_content=d) for d in data]

        # split documents by token_max minus 150 to take into account our prompt's token length
        text_splitter = TokenTextSplitter(chunk_size=self.token_max-150, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)

        return documents

    def run(self, data: List[Any]):
        docs = self._load_dataset(data)
        return self.reduce_documents_chain.combine_docs(docs, token_max=self.token_max)
