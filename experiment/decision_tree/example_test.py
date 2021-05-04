import os
import hydra
import logging
import pickle
import numpy as np
from .finite_cartpole import CartPoleRangeSet, default_range_set
from .solver import unpack_feature_list


def load_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data


class ExampleTester(object):
    def __init__(self, cfg):
        self._data = load_data(cfg.example_file)
        self._num_examples = cfg.num_examples_for_test
        if cfg.use_custom_range:
            self._range = CartPoleRangeSet(cfg.custom_ranges)
        else:
            self._range = default_range_set
        self._data_to_feature_list()
        if cfg.transform_example:
            self._set_value = dict()
        else:
            self._set_value = None

    def _data_to_feature_list(self):
        self._feature_action_list = []
        for data in self._data[:self._num_examples]:
            obs, action = data
            # print(obs, action)
            features = unpack_feature_list(self._range.convert_to_features(obs))
            self._feature_action_list += [(features, action)]
            # print(features, action)

    def check(self, verbose=True):
        print("Total number of examples = {}".format(len(self._feature_action_list)))
        for i in range(self._num_examples):
            count_equal = 0
            equals = []
            for j in range(self._num_examples):
                if i != j:
                    if self._feature_action_list[i][0] == self._feature_action_list[j][0] and \
                            self._feature_action_list[i][1] != self._feature_action_list[j][1]:
                        if verbose:
                            print(self._feature_action_list[i])
                            print(self._feature_action_list[j])
                            print(self._data[i][0] - self._data[j][0])
                        count_equal += 1
                        equals.append(j)
                        if i < j:
                            if i in self._set_value:
                                if j not in self._set_value:
                                    self._set_value[j] = self._set_value[i]
                            else:
                                if j in self._set_value:
                                    self._set_value[i] = self._set_value[j]
                                else:
                                    self._set_value[i] = self._data[i][1]
                                    self._set_value[j] = self._data[i][1]
            if count_equal > 0:
                if verbose:
                    print("data # {} has {} number of equal examples ({}).".format(i, count_equal, equals))

    def transform(self, filename):
        new_data = []
        for i in range(self._num_examples):
            if i in self._set_value:
                new_data.append((self._data[i][0], np.int64(self._set_value[i])))
            else:
                new_data.append(self._data[i])
        with open(filename, 'wb') as handle:
            pickle.dump(new_data, handle)
        print("saved file to {}".format(os.getcwd()))

    def save_feature_action_list(self, filename):
        with open(filename, 'w') as f:
            count = 0
            for features, action in self._feature_action_list:
                count += 1
                f.write(str(count) + ": ")
                for i, val in enumerate(features):
                    f.write("({}:{}) {}, ".format(i // 20 + 1, i % 20 + 1, val))
                f.write(str(action))
                f.write('\n')


@hydra.main(config_path="config", config_name="example_tester")
def main(cfg):
    logging.info("Working directory : {}".format(os.getcwd()))
    example_tester = ExampleTester(cfg.example_test)
    if cfg.example_test.perform_check:
        example_tester.check()
    if cfg.example_test.transform_example:
        example_tester.transform(cfg.example_test.save_filename)
    if cfg.example_test.save_feature_action_list:
        example_tester.save_feature_action_list(cfg.example_test.feature_action_list_filename)


if __name__ == "__main__":
    main()
