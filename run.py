import math
import sys

import numpy as np
import pandas as pd
from IPython import display

from helper import tf_basic_model

pd.options.display.max_rows = 100
pd.options.display.max_columns = 200
pd.options.display.float_format = '{:.1f}'.format

example_data_frame = pd.read_csv('./data/training.csv', sep=",")
testing_data_frame = pd.read_csv('./data/sorted_test.csv', sep=",")
example_data_frame = example_data_frame.reindex(np.random.permutation(example_data_frame.index))

# display.display(example_data_frame.describe())
# display.display(example_data_frame.columns)
# print (example_data_frame.isnull().sum())
#
# sys.exit(0)


training_dataframe = example_data_frame.head(958)
validation_dataframe = example_data_frame.tail(200)


training_examples = tf_basic_model.preprocess_features(training_dataframe)
training_targets = tf_basic_model.preprocess_targets(training_dataframe)

validation_examples = tf_basic_model.preprocess_features(validation_dataframe)
validation_targets = tf_basic_model.preprocess_targets(validation_dataframe)

testing_examples = tf_basic_model.preprocess_features(testing_data_frame)
testing_targets = tf_basic_model.preprocess_targets(testing_data_frame)

dnn_regressor = tf_basic_model.train_nn_regression_model(learning_rate=0.001,
                                                         steps=2000,
                                                         batch_size=100,
                                                         hidden_units=[10, 10],
                                                         training_examples=training_examples,
                                                         training_targets=training_targets,
                                                         validation_examples=validation_examples,
                                                         validation_targets=validation_targets)

prediction = tf_basic_model.submit_prediction(dnn_regressor, testing_examples, testing_targets)
