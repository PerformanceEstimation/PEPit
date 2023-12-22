import unittest

from PEPit.tools.dict_operations import merge_dict, prune_dict, multiply_dicts, symmetrize_dict


class TestDictOperations(unittest.TestCase):

    def setUp(self):
        self.dict1 = {'a': 5, 'b': 6, 'q': 11}
        self.dict2 = {'a': 2, 'b': 8, 'w': 0}

    def test_merge_dict(self):
        summed_dict = {'a': 7, 'b': 14, 'q': 11, 'w': 0}
        self.assertEqual(merge_dict(dict1=self.dict1, dict2=self.dict2), summed_dict)

    def test_multiply_dicts(self):

        product_dict = {('a', 'a'): 10, ('a', 'b'): 40, ('a', 'w'): 0,
                        ('b', 'a'): 12, ('b', 'b'): 48, ('b', 'w'): 0,
                        ('q', 'a'): 22, ('q', 'b'): 88, ('q', 'w'): 0,
                        }
        self.assertEqual(multiply_dicts(dict1=self.dict1, dict2=self.dict2), product_dict)

    def test_prune_dict(self):
        self.assertEqual(prune_dict(self.dict2), {'a': 2, 'b': 8})

    def test_symmetrize_dict(self):
        symmetric_product_dict = {'a': 5, 'b': 6, 'q': 11,
                                  ('a', 'a'): 10, ('a', 'b'): 26, ('a', 'w'): 0, ('w', 'a'): 0,
                                  ('b', 'a'): 26, ('b', 'b'): 48, ('b', 'w'): 0, ('w', 'b'): 0,
                                  ('q', 'a'): 11, ('q', 'b'): 44, ('q', 'w'): 0,
                                  ('a', 'q'): 11, ('b', 'q'): 44, ('w', 'q'): 0,
                                  }

        self.assertEqual(symmetrize_dict(merge_dict(self.dict1, multiply_dicts(dict1=self.dict1, dict2=self.dict2))),
                         symmetric_product_dict)
