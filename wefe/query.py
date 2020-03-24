import logging
import numpy as np
from itertools import combinations


class Query:

    def __init__(self, target_sets: list, attribute_sets: list, target_sets_names=None, attribute_sets_names=None):
        """A query object is a container for a pair of a attribute sets and t target sets. 
        
        Arguments:
            target_sets {list} -- An array with arrays of target sets. Each word set must represent a target word.
            attribute_sets {list} -- An array with arrays of attribute sets. Each word of any element must represent a target word.
        
        Keyword Arguments:
            target_sets_names {list} -- [description] (default: {None})
            attribute_sets_names {list} -- [description] (default: {None})
        """

        # check input type
        if (not isinstance(target_sets, (list, np.ndarray))):
            raise TypeError(
                "target_sets must be a list or a numpy array which contains lists or arrays of strings. Given: {}".
                format(type(target_sets)))

        if (not isinstance(attribute_sets, (list, np.ndarray))):
            raise TypeError(
                "attribute_sets must be a list or a numpy array which contains lists or arrays of strings. Given: {}".
                format(type(attribute_sets)))

        # check input array sizes
        if len(target_sets) == 0:
            raise Exception('target set must have at least one array of words. given: {}'.format(target_sets))
        if len(attribute_sets) == 0:
            raise Exception(
                'attribute_sets set must have at least one array of words. given: {}'.format(attribute_sets))

        for idx, target_set in enumerate(target_sets):
            if not isinstance(target_set, (list, np.ndarray)):
                raise TypeError("Each target set must be a list or an array of strings. Given: {} at postion {}".format(
                    type(target_set), idx))
            for word_idx, word in enumerate(target_set):
                if (not isinstance(word, str)):
                    raise TypeError('All elements in target set {} must be strings. Given: {} at position {}'.format(
                        idx, type(word), word_idx))

        for idx, attribute_set in enumerate(attribute_sets):
            if not isinstance(attribute_set, (list, np.ndarray, np.generic)):
                raise TypeError(
                    "Each attribute set must be a list or an array of strings. Given: {} at postion {}".format(
                        type(target_set), idx))
            for word_idx, word in enumerate(attribute_set):
                if (not isinstance(word, str)):
                    raise TypeError('All elements in attribute set {} must be strings. Given: {} at position {}'.format(
                        idx, type(word), word_idx))

        # set target and attributes sets to this instance.
        self.target_sets = target_sets
        self.attribute_sets = attribute_sets

        # set the template/cardinality (t, a) of the sets
        self.template = (len(target_sets), len(attribute_sets))

        # set target sets names.
        if isinstance(target_sets_names, type(None)):
            self.target_sets_names = ["Target set {}".format(i) for i in range(self.template[0])]
        else:
            if (len(target_sets_names) != self.template[0]):
                logging.warning(
                    'target_sets_names does not have the same elements ({}) as target_sets ({}). Setting default names'.
                    format(len(target_sets_names), self.template[0]))
                self.target_sets_names = ["Target set {}".format(i) for i in range(self.template[0])]
            else:
                self.target_sets_names = target_sets_names

        # set attribute and attribute sets names.
        if isinstance(attribute_sets_names, type(None)):
            self.attribute_sets_names = ["Attribute set {}".format(i) for i in range(self.template[1])]
        else:
            if (len(attribute_sets_names) != self.template[1]):
                logging.warning(
                    'attribute_sets_names does not have the same elements ({}) as attribute_sets ({}). Setting default names'
                    .format(len(attribute_sets_names), self.template[1]))
                self.attribute_sets_names = ["Attribute set {}".format(i) for i in range(self.template[1])]

            else:
                self.attribute_sets_names = attribute_sets_names

        self.query_name = self._generate_query_name()

    def __eq__(self, other):

        if not isinstance(other, Query):
            return False

        if self.template[0] != other.template[0]:
            return False
        if self.template[1] != other.template[1]:
            return False

        for target_set, other_target_set in zip(self.target_sets, other.target_sets):
            if target_set != other_target_set:
                return False

        for attribute_set, other_attribute_set in zip(self.attribute_sets, other.attribute_sets):
            if attribute_set != other_attribute_set:
                return False

        for names, other_names in zip(self.target_sets_names, other.target_sets_names):
            if names != other_names:
                return False

        for names, other_names in zip(self.attribute_sets_names, other.attribute_sets_names):
            if names != other_names:
                return False
        return True

    def generate_subqueries(self, new_template: tuple):
        """Generate the subqueries from this query using the given template
        
        Parameters
        ----------
        new_template : tuple
            [description]
        
        Returns
        -------
        [type]
            [description]
        
        Raises
        ------
        TypeError
            [description]
        TypeError
            [description]
        Exception
            [description]
        Exception
            [description]
        """

        if not isinstance(new_template[0], int):
            raise TypeError('The new target cardinality (new_template[0]) must be int. Given: {}'.format(
                new_template[0]))
        if not isinstance(new_template[1], int):
            raise TypeError('The new attribute cardinality (new_template[1]) must be int. Given: {}'.format(
                new_template[1]))

        if new_template[0] > self.template[0]:
            raise Exception(
                'The new target cardinality (new_template[0]) must be equal or less than the original target set cardinality. Given: {}'
                .format(new_template[0]))
        if new_template[1] > self.template[1]:
            raise Exception(
                'The new attribute cardinality (new_template[1]) must be equal or less than the original attribute set cardinality. Given: {}'
                .format(new_template[1]))

        target_combinations = list(combinations(range(self.template[0]), new_template[0]))
        attribute_combinations = list(combinations(range(self.template[1]), new_template[1]))

        target_subsets = [[self.target_sets[idx] for idx in combination] for combination in target_combinations]
        target_subsets_names = [
            [self.target_sets_names[idx] for idx in combination] for combination in target_combinations
        ]
        attribute_subsets = [[self.attribute_sets[idx] for idx in combination] for combination in attribute_combinations
                            ]
        attribute_subsets_names = [
            [self.attribute_sets_names[idx] for idx in combination] for combination in attribute_combinations
        ]

        subqueries = [[
            Query(target_subset, attribute_subset, target_subset_name, attribute_subset_name)
            for attribute_subset, attribute_subset_name in zip(attribute_subsets, attribute_subsets_names)
        ]
                      for target_subset, target_subset_name in zip(target_subsets, target_subsets_names)]

        return np.array(subqueries).flatten().tolist()

    def _generate_query_name(self) -> str:
        """Generates the query name from the name of its target and attribute sets.
        
        Parameters
        ----------
        query : Query
            The query to be tested.
        
        Returns
        -------
        str
            The name of the query.
        """

        target_sets_names = self.target_sets_names
        attribute_sets_names = self.attribute_sets_names

        if len(target_sets_names) == 1:
            target = target_sets_names[0]
        elif len(target_sets_names) == 2:
            target = target_sets_names[0] + " and " + target_sets_names[1]
        else:
            target = ', '.join([str(x) for x in target_sets_names[0:-1]]) + ' and ' + target_sets_names[-1]

        if len(attribute_sets_names) == 1:
            attribute = attribute_sets_names[0]
        elif len(attribute_sets_names) == 2:
            attribute = attribute_sets_names[0] + " and " + attribute_sets_names[1]
        else:
            attribute = ', '.join([str(x) for x in attribute_sets_names[0:-1]]) + ' and ' + attribute_sets_names[-1]

        return target + ' wrt ' + attribute
