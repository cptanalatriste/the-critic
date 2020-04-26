from collections import Counter


def get_word_count(reviews):
    all_words = [word for review in reviews for word in review]
    return Counter(all_words)


def sort_by_frequency(frequency_map):
    return sorted(frequency_map.keys(), key=lambda key: frequency_map[key], reverse=True)
