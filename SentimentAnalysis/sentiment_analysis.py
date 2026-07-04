"""Sentiment analysis over text using the Watson NLP BERT SentimentPredict service."""
import requests
import json


def sentiment_analyzer(text_to_analyze):
    """Run sentiment analysis on the given text and return a dict of results.

    Returns a dict with keys label (e.g. 'SENT_POSITIVE') and score. Both
    are None when the Watson NLP service does not return a 200 response
    (e.g. status code 500 for text it cannot interpret).
    """
    url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'
    myobj = {"raw_document": {"text": text_to_analyze}}
    header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}

    response = requests.post(url, json=myobj, headers=header)

    if response.status_code == 200:
        formatted_response = json.loads(response.text)
        label = formatted_response['documentSentiment']['label']
        score = formatted_response['documentSentiment']['score']
    else:
        label = None
        score = None

    return {'label': label, 'score': score}
