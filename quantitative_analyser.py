import concurrent

from comment_sentiment_classifier import CommentSentimentClassifier, CommentSentimentResponse
from toxic_message_detector import ToxicMessageDetector, ToxicMessageResponse
from typing import List


class QuantitativeAnalysisResponse:
    positive_comments_total: int
    negative_comments_total: int
    rejected_comments_indexes: List[int]

    def __init__(self):
        self.positive_comments_total = 0
        self.negative_comments_total = 0
        self.rejected_comments_indexes = []


class QuantitativeAnalyser:
    def __init__(self):
        self.comment_sentiment_classifier = CommentSentimentClassifier()
        self.toxic_message_detector = ToxicMessageDetector()

    def __call__(self, texts: list[str], thread_number: int = 30, batch_size: int = 33) -> QuantitativeAnalysisResponse:
        executor = concurrent.futures.ThreadPoolExecutor(thread_number)
        futures = [executor.submit(self._inference, i, texts[i:i + batch_size]) for i in
                   range(0, len(texts), batch_size)]
        concurrent.futures.wait(futures)

        quantitative_analysis_response = QuantitativeAnalysisResponse()
        for future in futures:
            result = future.result()
            quantitative_analysis_response.rejected_comments_indexes.extend(result.rejected_comments_indexes)
            quantitative_analysis_response.positive_comments_total += result.positive_comments_total
            quantitative_analysis_response.negative_comments_total += result.negative_comments_total

        return quantitative_analysis_response

    def _inference(self, first_index: int, texts: list[str]) -> QuantitativeAnalysisResponse:
        quantitative_analysis_response: QuantitativeAnalysisResponse = QuantitativeAnalysisResponse()
        current_index = first_index
        for text in texts:
            self._callback_start_inference(current_index, text)
            toxic_message_response: ToxicMessageResponse = self.toxic_message_detector(text)
            if toxic_message_response.is_toxic():
                quantitative_analysis_response.rejected_comments_indexes.append(current_index)
                self._callback_end_inference(current_index, toxic_message_response)
            else:
                comment_sentiment_response: CommentSentimentResponse = self.comment_sentiment_classifier(text)
                if comment_sentiment_response.is_positive:
                    quantitative_analysis_response.positive_comments_total += 1
                else:
                    quantitative_analysis_response.negative_comments_total += 1
                self._callback_end_inference(current_index, toxic_message_response, comment_sentiment_response)
            current_index += 1
        return quantitative_analysis_response

    def _callback_start_inference(self, index, value):
        print(f"start inference for index: {index} and value: {value[0:50]}...")

    def _callback_end_inference(self, index, toxic_message_response: ToxicMessageResponse,
                                comment_sentiment_response: CommentSentimentResponse = None):
        message: str = f"Element at index {index} is {'not' if not toxic_message_response.is_toxic() else ''} toxic. "
        if comment_sentiment_response:
            message += f"Sentiment classification returned {'positive' if comment_sentiment_response.is_positive else 'negative'}."
        print(message)
        if toxic_message_response.is_toxic():
            print(f"Scores: {toxic_message_response}")
