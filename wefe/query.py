import logging
import numpy as np
from itertools import combinations
from typing import Union


class Query:
    """A container for attribute and target word sets."""
    def __init__(self, target_sets: Union[np.ndarray, list],
                 attribute_sets: Union[np.ndarray, list],
                 target_sets_names: Union[np.ndarray, list] = None,
                 attribute_sets_names: Union[np.ndarray, list] = None):
        """Initializes the container. It could include a name for each word set.
        
        Parameters
        ----------
        target_sets : Union[np.ndarray, list]
            Array or list that contain the target word sets.
        attribute_sets : Union[np.ndarray, Iterable]
            Array or list that contain the attribute word sets.
        target_sets_names : Union[np.ndarray, Iterable], optional
            Array or list that contain the word sets names, by default None
        attribute_sets_names : Union[np.ndarray, Iterable], optional
            Array or list that contain the attribute sets names, by default None

        Attributes
        ----------
        target_sets_ : list
            Array or list with the lists of target words.
        attribute_sets_ : list
            Array or list with the lists of target words.
        template_ : tuple
            A tuple that contains the template: the number of the target and attribute sets respectively.
        target_sets_names_ : list
            Array or list with the names of target sets.
        attribute_sets_names_ : list
            Array or list with the lists of target words.
        query_name_ : str
            A string that contains the auto-generated name of the query.

        Raises
        ------
        TypeError
            if target_sets are not a iterable or np.ndarray instance.
        TypeError
            if attribute_sets are not a iterable or np.ndarray instance.
        Exception
            if the length of target_sets is 0.
        TypeError
            if some element of target_sets is not an array or list.
        TypeError
            if some element of some target set is not an string.
        TypeError
            if some element of attribute_sets is not an array or list.
        TypeError
            if some element of some attribute set is not an string.

        Examples
        --------
        Construct a Query with 2 sets of target words and one set of attribute words.

        >>> male_terms = ['male', 'man', 'boy']
        >>> female_terms = ['female', 'woman', 'girl']
        >>> science_terms = ['science','technology','physics']
        >>> query = Query([male_terms, female_terms], [science_terms], ['Male terms', 'Female terms'], ['Science terms'])
        >>> query.target_sets_
        [['male', 'man', 'boy'], ['female', 'woman', 'girl']]
        >>> query.attribute_sets_
        [['science', 'technology', 'physics']]
        >>> query.query_name_
        'Male terms and Female terms wrt Science terms'
        """

        # check input type
        if (not isinstance(target_sets, (list, np.ndarray))):
            raise TypeError(
                "target_sets must be a numpy array or list. Given: {}".format(
                    type(target_sets)))

        if (not isinstance(attribute_sets, (list, np.ndarray))):
            raise TypeError(
                "attribute_sets must be a numpy array or list. Given: {}".
                format(type(attribute_sets)))

        # check input array sizes
        if len(target_sets) == 0:
            raise Exception(
                'target_sets must have at least one array or list of words. given: {}'
                .format(target_sets))

        # check all words that target sets contains.
        for idx, target_set in enumerate(target_sets):
            if not isinstance(target_set, (np.ndarray, list)):
                raise TypeError(
                    "Each target set must be a list or an array of strings. Given: {} at postion {}"
                    .format(type(target_set), idx))
            for word_idx, word in enumerate(target_set):
                if (not isinstance(word, str)):
                    raise TypeError(
                        'All elements in target set {} must be strings. Given: {} at position {}'
                        .format(idx, type(word), word_idx))

        # check all words that attribute sets contains.
        for idx, attribute_set in enumerate(attribute_sets):
            if not isinstance(attribute_set, (np.ndarray, list)):
                raise TypeError(
                    "Each attribute set must be a list or an array of strings. Given: {} at postion {}"
                    .format(type(target_set), idx))
            for word_idx, word in enumerate(attribute_set):
                if (not isinstance(word, str)):
                    raise TypeError(
                        'All elements in attribute set {} must be strings. Given: {} at position {}'
                        .format(idx, type(word), word_idx))

        # set target and attributes sets to this instance.
        self.target_sets_ = target_sets
        self.attribute_sets_ = attribute_sets

        # set the template/cardinality (t, a) of the sets
        self.template_ = (len(target_sets), len(attribute_sets))

        # set target sets names.
        if isinstance(target_sets_names, type(None)):
            self.target_sets_names_ = [
                "Target set {}".format(i) for i in range(self.template_[0])
            ]
        else:
            if (len(target_sets_names) != self.template_[0]):
                logging.warning(
                    'target_sets_names does not have the same elements ({}) as target_sets ({}). Setting default names'
                    .format(len(target_sets_names), self.template_[0]))
                self.target_sets_names_ = [
                    "Target set {}".format(i) for i in range(self.template_[0])
                ]
            else:
                self.target_sets_names_ = target_sets_names

        # set attribute and attribute sets names.
        if isinstance(attribute_sets_names, type(None)):
            self.attribute_sets_names_ = [
                "Attribute set {}".format(i) for i in range(self.template_[1])
            ]
        else:
            if (len(attribute_sets_names) != self.template_[1]):
                logging.warning(
                    'attribute_sets_names does not have the same elements ({}) as attribute_sets ({}). Setting default names'
                    .format(len(attribute_sets_names), self.template_[1]))
                self.attribute_sets_names_ = [
                    "Attribute set {}".format(i)
                    for i in range(self.template_[1])
                ]

            else:
                self.attribute_sets_names_ = attribute_sets_names

        self.query_name_ = self._generate_query_name()

    def __eq__(self, other):

        if not isinstance(other, Query):
            return False

        if self.template_[0] != other.template_[0]:
            return False
        if self.template_[1] != other.template_[1]:
            return False

        for target_set, other_target_set in zip(self.target_sets_,
                                                other.target_sets_):
            if target_set != other_target_set:
                return False

        for attribute_set, other_attribute_set in zip(self.attribute_sets_,
                                                      other.attribute_sets_):
            if attribute_set != other_attribute_set:
                return False

        for names, other_names in zip(self.target_sets_names_,
                                      other.target_sets_names_):
            if names != other_names:
                return False

        for names, other_names in zip(self.attribute_sets_names_,
                                      other.attribute_sets_names_):
            if names != other_names:
                return False
        return True

    def get_subqueries(self, new_template: tuple):
        """Generate the subqueries from this query using the given template
        """

        if not isinstance(new_template[0], int):
            raise TypeError(
                'The new target cardinality (new_template[0]) must be int. Given: {}'
                .format(new_template[0]))
        if not isinstance(new_template[1], int):
            raise TypeError(
                'The new attribute cardinality (new_template[1]) must be int. Given: {}'
                .format(new_template[1]))

        if new_template[0] > self.template_[0]:
            raise Exception(
                'The new target cardinality (new_template[0]) must be equal or less than the original target set cardinality. Given: {}'
                .format(new_template[0]))
        if new_template[1] > self.template_[1]:
            raise Exception(
                'The new attribute cardinality (new_template[1]) must be equal or less than the original attribute set cardinality. Given: {}'
                .format(new_template[1]))

        target_combinations = list(
            combinations(range(self.template_[0]), new_template[0]))
        attribute_combinations = list(
            combinations(range(self.template_[1]), new_template[1]))

        target_subsets = [[self.target_sets_[idx] for idx in combination]
                          for combination in target_combinations]
        target_subsets_names = [[
            self.target_sets_names_[idx] for idx in combination
        ] for combination in target_combinations]
        attribute_subsets = [[
            self.attribute_sets_[idx] for idx in combination
        ] for combination in attribute_combinations]
        attribute_subsets_names = [[
            self.attribute_sets_names_[idx] for idx in combination
        ] for combination in attribute_combinations]

        subqueries = [[
            Query(target_subset, attribute_subset, target_subset_name,
                  attribute_subset_name)
            for attribute_subset, attribute_subset_name in zip(
                attribute_subsets, attribute_subsets_names)
        ]
                      for target_subset, target_subset_name in zip(
                          target_subsets, target_subsets_names)]

        return np.array(subqueries).flatten().tolist()

    def _generate_query_name(self) -> str:
        """Generates the query name from the name of its target and attribute sets.
        
        Returns
        -------
        str
            The name of the query.
        """

        target_sets_names = self.target_sets_names_
        attribute_sets_names = self.attribute_sets_names_

        if len(target_sets_names) == 1:
            target = target_sets_names[0]
        elif len(target_sets_names) == 2:
            target = target_sets_names[0] + " and " + target_sets_names[1]
        else:
            target = ', '.join([str(x) for x in target_sets_names[0:-1]
                                ]) + ' and ' + target_sets_names[-1]

        if len(attribute_sets_names) == 0:
            return target

        if len(attribute_sets_names) == 1:
            attribute = attribute_sets_names[0]
        elif len(attribute_sets_names) == 2:
            attribute = attribute_sets_names[
                0] + " and " + attribute_sets_names[1]
        else:
            attribute = ', '.join([str(x) for x in attribute_sets_names[0:-1]
                                   ]) + ' and ' + attribute_sets_names[-1]

        return target + ' wrt ' + attribute
