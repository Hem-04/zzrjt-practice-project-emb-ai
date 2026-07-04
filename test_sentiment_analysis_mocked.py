"""Local sanity check for sentiment_analyzer logic, without hitting the
real Watson NLP service (that endpoint only resolves inside the IBM
Skills Network lab environment). Not part of the graded test suite."""
import json
import unittest
from unittest.mock import patch
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer


def fake_response(status_code, label=None, score=None):
    class FakeResponse:
        pass
    resp = FakeResponse()
    resp.status_code = status_code
    resp.text = json.dumps({'documentSentiment': {'label': label, 'score': score}})
    return resp


class TestSentimentAnalysisMocked(unittest.TestCase):
    @patch('SentimentAnalysis.sentiment_analysis.requests.post')
    def test_positive_label(self, mock_post):
        mock_post.return_value = fake_response(200, 'SENT_POSITIVE', 0.99)
        result = sentiment_analyzer("I love working with Python")
        self.assertEqual(result['label'], 'SENT_POSITIVE')
        self.assertEqual(result['score'], 0.99)

    @patch('SentimentAnalysis.sentiment_analysis.requests.post')
    def test_invalid_input_returns_none(self, mock_post):
        mock_post.return_value = fake_response(500)
        result = sentiment_analyzer("as987da-6s2d aweadsa")
        self.assertIsNone(result['label'])
        self.assertIsNone(result['score'])


if __name__ == '__main__':
    unittest.main()
