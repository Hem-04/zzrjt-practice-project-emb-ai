import unittest
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer


class TestSentimentAnalyzer(unittest.TestCase):
    def test_sentiment_analyzer(self):
        result = sentiment_analyzer("I love working with Python")
        self.assertEqual(result['label'], 'SENT_POSITIVE')

        result = sentiment_analyzer("I hate working with Python")
        self.assertEqual(result['label'], 'SENT_NEGATIVE')

        result = sentiment_analyzer("I am neutral on Python")
        self.assertEqual(result['label'], 'SENT_NEUTRAL')


unittest.main()
