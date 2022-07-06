"""
Functions for explaining text classifiers.
"""
from functools import partial
import itertools
import json
import re

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

from . import explanation
from . import lime_base


class TextDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.indexed_string.word(x[0]),
                '-'.join(
                    map(str,
                        self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)
        exp = [(self.indexed_string.word(x[0]),
                self.indexed_string.string_position(x[0]),
                x[1]) for x in exp]
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class IndexedString(object):
    """String with various indexes."""

    def __init__(self, raw_string, split_expression=r'\W+', bow=True,
                 mask_string=None):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        self.raw = raw_string
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            # with the split_expression as a non-capturing group (?:), we don't need to filter out
            # the separator character from the split results.
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class IndexedCharacters(object):
    """String with various indexes."""

    def __init__(self, raw_string, bow=True, mask_string=None):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            bow: if True, a char is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same character. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace characters with this if bow=False
                if None, default value is chr(0)
        """
        self.raw = raw_string
        self.as_list = list(self.raw)
        self.as_np = np.array(self.as_list)
        self.mask_string = chr(0) if mask_string is None else mask_string
        self.string_start = np.arange(len(self.raw))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, char in enumerate(self.as_np):
            if char in non_vocab:
                continue
            if bow:
                if char not in vocab:
                    vocab[char] = len(vocab)
                    self.inverse_vocab.append(char)
                    self.positions.append([])
                idx_char = vocab[char]
                self.positions[idx_char].append(i)
            else:
                self.inverse_vocab.append(char)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join(
                [self.as_list[i] if mask[i] else self.mask_string
                 for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class LimeTextExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False,
                 pair=False,
                 mode="classification"):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level
        """
        Mudança
        """
        self.pair = pair
        self.mode = mode

        """
        Término mudança
        """

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None
                         ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Transformei processo em função para usar duas vezes no caso de pares como input.
        def indexed_string_text(text_instance):
            indexed_string = (IndexedCharacters(
                text_instance, bow=self.bow, mask_string=self.mask_string)
                              if self.char_level else
                              IndexedString(text_instance, bow=self.bow,
                                            split_expression=self.split_expression,
                                            mask_string=self.mask_string))
            return indexed_string

        """
        Mudança
        """
        # Ajuste do index_string para dois textos no caso de pares como input
        if self.pair == True:
            try:
                pair = text_instance.split('[SEP]')
                text1 = pair[0]
                text2 = pair[1]
            except:
                raise TypeError("Sentences must be separated by [SEP] token.")
            indexed_string = (indexed_string_text(text1),indexed_string_text(text2))
        else:
            indexed_string = indexed_string_text(text_instance)
        """
        Término a mudança
        """

        domain_mapper = TextDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state,
                                          mode=self.mode)
        ret_exp.predict_proba = yss[0]
        """
        Mudança
        """

        if self.mode == "regression":
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]


        # Necessário mudar variavel labels no caso de ser regressão
        #if self.mode == "regression":
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        else:
        #Término mudança
        # Se for classificação
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        """
        Mudança
        """
        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
        """
        Término mudança
        """
        return ret_exp

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        def neighborhood_text_samples(indexed_string):
            doc_size = indexed_string.num_words()
            sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)
            data = np.ones((num_samples, doc_size))
            data[0] = np.ones(doc_size)
            features_range = range(doc_size)
            inverse_data = [indexed_string.raw_string()]
            for i, size in enumerate(sample, start=1):
                inactive = self.random_state.choice(features_range, size,
                                                    replace=False)
                data[i, inactive] = 0
                inverse_data.append(indexed_string.inverse_removing(inactive))
            return inverse_data, data

        if self.pair == True:
            # Procedimento para dois textos
            try:
                inverse_data1,data1 = neighborhood_text_samples(indexed_string[0])
                inverse_data2,data2 = neighborhood_text_samples(indexed_string[1])
                inverse_data = [t1+'[SEP]'+ for i, j in zip(inverse_data1,inverse_data2)]
                data = np.concatenate((data1, data2), axis=1)
            except:
                raise TypeError("Problema no index_string")
        else:
            # Entrada de um texto apenas, continua normal
            inverse_data, data = neighborhood_text_samples(indexed_string)

        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances
