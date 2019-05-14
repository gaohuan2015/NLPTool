import unittest
import Preprocess as pre


class TestPrepareFunc(unittest.TestCase):
    def test_loaddic(self):
        path = '/Data/words2id.json'
        dic = pre.load_dic_from_json(path)
        self.assertEqual(len(dic), 90760)

    def test_loadwordembedding(self):
        path = '/Data/words2id.json'
        dic = pre.load_wordembedding_from_json(path)
        self.assertEqual(len(dic), 90760)

    def test_loaddata(self):
        path = '/Data/train.json'
        dic = pre.load_data_from_json(path)
        self.assertEqual(len(dic), 3)
        self.assertEqual(len(dic[0]), 56195)

if __name__ == "__main__":
    unittest.main()