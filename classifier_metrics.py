from comment_sentiment_classifier import CommentSentimentClassifier, CommentSentimentResponse
from data_preprocessor import DataPreprocessor
from sklearn.metrics import precision_recall_fscore_support

data_preprocessor = DataPreprocessor("Exemple 4/cetace.csv")
df = data_preprocessor.preprocess().head(10)
comment_sentiment_classifier: CommentSentimentClassifier = CommentSentimentClassifier("mtcet-nlp")
results = []
for index, row in df.iterrows():
    elems = row['Non pertinent;Favorable;Défavorable;"objet"'].split(";")
    if elems[0] != "1":
        comment_sentiment_response: CommentSentimentResponse = comment_sentiment_classifier(row["whole_text"])
        results.append(1 if comment_sentiment_response.is_positive else 0)


def f(row):
    elems = row['Non pertinent;Favorable;Défavorable;"objet"'].split(";")
    if elems[0] == "1":
        return -1
    if elems[1] == "1":
        return 1
    return 0


df['classification'] = df.apply(f, axis=1)
df = df.drop(df[df.classification == -1].index)

result = precision_recall_fscore_support(df["classification"].to_list(), results)

