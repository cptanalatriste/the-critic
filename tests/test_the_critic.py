import unittest

from the_critic import get_word_count, sort_by_frequency


class TestTheCritic(unittest.TestCase):

    def test_sorting_words(self):
        a_review = "this is bad"
        another_review = "this is really bad"
        and_another_one = "bad"

        reviews = [review.split() for review in [a_review, another_review, and_another_one]]
        word_count = get_word_count(reviews=reviews)

        self.assertEqual(2, word_count["this"])
        self.assertEqual(1, word_count["really"])

        sorted_words = sort_by_frequency(word_count)
        self.assertEqual("bad", sorted_words[0])
        self.assertEqual("really", sorted_words[-1])
