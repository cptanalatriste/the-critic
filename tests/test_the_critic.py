import copy
import unittest

import torch
from torch import optim

from the_critic import get_word_count, sort_by_frequency, train_network
from train.model import LSTMClassifier


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

    def test_train_network(self):
        model = LSTMClassifier(embedding_dim=32, hidden_dim=100, vocab_size=500)
        optimizer = optim.Adam(params=model.parameters())
        loss_fn = torch.nn.BCELoss()

        features = torch.randint(low=0, high=500, size=(50, 501))
        labels = torch.randint(low=0, high=1, size=(50,))

        parameters_before_training = copy.deepcopy(model.state_dict())
        loss = train_network(model=model, criterion=loss_fn, features=features, labels=labels, optimiser=optimizer)
        parameters_after_training = copy.deepcopy(model.state_dict())

        self.assertIsNotNone(loss.data.item())
        self.assertFalse(
            compare_model_parameters(parameters_before_training, parameters_after_training))


def compare_model_parameters(parameters, more_parameters):
    """
    Taken from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    :param parameters:
    :param more_parameters:
    :return:
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(parameters.items(), more_parameters.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                return False
    if models_differ == 0:
        return True
