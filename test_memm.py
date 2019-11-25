import itertools
import pytest
import numpy as np

import memm


PTB_TAGS = {
    "#", "$", "''", "``", ",", "-LRB-", "-RRB-", ".", ":", "CC", "CD", "DT",
    "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS",
    "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
    "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
}


def test_read_ptbtagged():
    # keep a counter here (instead of enumerate) in case the iterator is empty
    token_count = 0
    sentence_count = 0
    for sentence in memm.read_ptbtagged("PTBSmall/train.tagged"):
        assert len(sentence) == 2
        tokens, pos_tags = sentence
        assert len(tokens) == len(pos_tags)
        assert all(pos in PTB_TAGS for pos in pos_tags)
        token_count += len(tokens)
        sentence_count += 1
    assert token_count == 191969
    assert sentence_count == 8020

    # check the sentence count in the dev set too
    assert sum(1 for _ in memm.read_ptbtagged("PTBSmall/dev.tagged")) == 5039


def test_train_tensors():
    classifier = memm.Classifier()
    ptb_train = memm.read_ptbtagged("PTBSmall/train.tagged")
    ptb_train = itertools.islice(ptb_train, 2)  # just the 1st 2 sentences
    features_matrix, labels_vector = classifier.train(ptb_train)
    assert features_matrix.shape[0] == 31
    assert labels_vector.shape[0] == 31

    # train.tagged starts with
    # Pierre	NNP
    # Vinken	NNP
    # ,	,
    # 61	CD
    # years	NNS
    # old	JJ
    assert features_matrix[4, classifier.feature_index("token=years")] == 1
    assert features_matrix[4, classifier.feature_index("token=old")] == 0
    assert features_matrix[4, classifier.feature_index("pos-1=CD")] == 1
    assert features_matrix[4, classifier.feature_index("pos-1=NNS")] == 0
    assert features_matrix[0, classifier.feature_index("pos-1=<s>")] == 1
    assert labels_vector[3] == classifier.label_index("CD")
    assert labels_vector[4] == classifier.label_index("NNS")


def test_predict_greedy(capsys):
    classifier = memm.Classifier()
    ptb_train = memm.read_ptbtagged("PTBSmall/train.tagged")
    ptb_train = itertools.islice(ptb_train, 2)  # just the 1st 2 sentences
    classifier.train(ptb_train)

    tokens = "Vinken is a director .".split()
    features_matrix, pos_tags = classifier.predict_greedy(tokens)

    # check that there is one feature vector per POS tag
    assert features_matrix.shape[0] == len(pos_tags)

    # check that all POS tags are in the PTB tagset
    assert all(pos_tag in PTB_TAGS for pos_tag in pos_tags)

    def last_pos_index(ptb_tag):
        return classifier.feature_index("pos-1=" + ptb_tag)

    # check that the first word ("The") has no pos-1 feature
    for ptb_tag in {"NNP", ",", "CD", "NNS", "JJ", "MD", "VB", "DT", "NN", "IN",
                    "VBZ", "VBG"}:
        assert features_matrix[0, last_pos_index(ptb_tag)] == 0

    # check that the remaining words have the correct pos-1 features
    for i, pos_tag in enumerate(pos_tags[:-1]):
        assert features_matrix[i + 1, last_pos_index(pos_tag)] > 0


def test_accuracy(capsys):
    classifier = memm.Classifier()
    ptb_train = memm.read_ptbtagged("PTBSmall/train.tagged")
    classifier.train(ptb_train)

    total_count = 0.0
    correct_count = 0
    ptb_dev = memm.read_ptbtagged("PTBSmall/dev.tagged")
    ptb_dev = itertools.islice(ptb_dev, 100)  # just the 1st 100 sentences
    for tokens, pos_tags in ptb_dev:
        total_count += len(tokens)
        predicted_tags = classifier.predict(tokens)
        assert len(predicted_tags) == len(pos_tags)
        for predicted_tag, true_tag in zip(predicted_tags, pos_tags):
            if predicted_tag == true_tag:
                correct_count += 1
    accuracy = correct_count / total_count

    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} accuracy on first 100 sentences of PTB dev"
            print(msg.format(accuracy))

    assert accuracy >= 0.93


@pytest.mark.xfail
def test_predict_viterbi():
    classifier = memm.Classifier()
    ptb_train = memm.read_ptbtagged("PTBSmall/train.tagged")
    ptb_train = itertools.islice(ptb_train, 2)  # just the 1st 2 sentences
    classifier.train(ptb_train)

    # POS tags in first 2 sentences
    possible_tags = {"NNP", ",", "CD", "NNS", "JJ", "MD", "VB", "DT", "NN",
                     "IN", ".", "VBZ", "VBG"}
    n_tags = len(possible_tags)

    # sample sentence to be fed to classifier
    tokens = "Vinken is a director .".split()
    trans_probs, viterbi_lattice, pos_tags = classifier.predict_viterbi(tokens)

    # check that the transition probabilities are the right shape
    # second axis is +1 since the last entry is transitions starting at <s>
    assert trans_probs.shape == (len(tokens), n_tags + 1, n_tags)

    # check that probability distribution from <s> to first word sums to 1
    s_index = n_tags
    prob_dist_sums = np.sum(np.exp(trans_probs), axis=-1)
    np.testing.assert_almost_equal(prob_dist_sums[0, s_index], 1)

    # check that probability distributions of all non-<s> pairs sum to 1
    for i in range(1, len(tokens)):
        for j in range(0, n_tags):
            np.testing.assert_almost_equal(prob_dist_sums[i, j], 1)

    # check that the lattice is the right shape
    assert viterbi_lattice.shape == (len(tokens), n_tags)

    # check that the numbers are not all the same in the lattice
    assert np.std(viterbi_lattice) > 0
    assert np.all(np.std(viterbi_lattice, axis=0) > 0)
    assert np.all(np.std(viterbi_lattice, axis=1) > 0)

    # check that the probabilities from <s> are on the first token
    np.testing.assert_almost_equal(viterbi_lattice[0], trans_probs[0, s_index])

    # check some probability calculations in the lattice
    np.testing.assert_almost_equal(viterbi_lattice[1, 2], max([
        viterbi_lattice[0, k] + trans_probs[1, k, 2] for k in range(n_tags)]))
    np.testing.assert_almost_equal(viterbi_lattice[3, 1], max([
        viterbi_lattice[2, k] + trans_probs[3, k, 1] for k in range(n_tags)]))
    np.testing.assert_almost_equal(viterbi_lattice[-1, 9], max([
        viterbi_lattice[-2, k] + trans_probs[-1, k, 9] for k in range(n_tags)]))

    # check that the POS tags are all valid tags
    assert all([pos_tag in PTB_TAGS for pos_tag in pos_tags])

    # check that the lattice's score for the predicted POS tag path matches
    # the score we would get from the transition probabilities
    pos_indexes = [classifier.label_index(t) for t in pos_tags]
    np.testing.assert_almost_equal(
        viterbi_lattice[-1, pos_indexes[-1]],
        sum(trans_probs[i, pos_indexes[i - 1] if i else s_index, pos_index]
            for i, pos_index in enumerate(pos_indexes)))

    # check that the selected POS path has the highest score of all the possible
    # paths through the lattice
    np.testing.assert_almost_equal(
        viterbi_lattice[-1, pos_indexes[-1]],
        max(trans_probs[0, s_index, index1] +
            trans_probs[1, index1, index2] +
            trans_probs[2, index2, index3] +
            trans_probs[3, index3, index4] +
            trans_probs[4, index4, index5]
            for index1 in range(n_tags)
            for index2 in range(n_tags)
            for index3 in range(n_tags)
            for index4 in range(n_tags)
            for index5 in range(n_tags)))
