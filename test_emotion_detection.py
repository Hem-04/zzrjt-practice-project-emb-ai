import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    def test_emotion_detector(self):
        result = emotion_detector("I am so happy I am doing this")
        self.assertEqual(result['dominant_emotion'], 'joy')

        result = emotion_detector("I am so angry I am doing this")
        self.assertEqual(result['dominant_emotion'], 'anger')

        result = emotion_detector("I am so disgusted I am doing this")
        self.assertEqual(result['dominant_emotion'], 'disgust')

        result = emotion_detector("I am so sad I am doing this")
        self.assertEqual(result['dominant_emotion'], 'sadness')

        result = emotion_detector("I am so afraid I am doing this")
        self.assertEqual(result['dominant_emotion'], 'fear')


unittest.main()
