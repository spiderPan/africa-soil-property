import math
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display

from helper import tf_basic_model

pd.options.display.max_rows = 100
pd.options.display.max_columns = 200
pd.options.display.float_format = '{:.1f}'.format

example_data_frame = pd.read_csv('./data/training.csv', sep=",")
testing_data_frame = pd.read_csv('./data/sorted_test.csv', sep=",")

example_data_frame = example_data_frame.reindex(np.random.permutation(example_data_frame.index))

training_dataframe = example_data_frame.head(958)
validation_dataframe = example_data_frame.tail(200)


training_examples = tf_basic_model.preprocess_features(training_dataframe)
training_targets = tf_basic_model.preprocess_targets(training_dataframe)

validation_examples = tf_basic_model.preprocess_features(validation_dataframe)
validation_targets = tf_basic_model.preprocess_targets(validation_dataframe)

testing_examples = tf_basic_model.preprocess_features(testing_data_frame)
testing_targets = tf_basic_model.preprocess_targets(testing_data_frame)

gradient_regressor, gradient_training_looses, gradient_validation_losses = tf_basic_model.train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.07),
    steps=10000,
    batch_size=50,
    hidden_units=[2048, 512, 32],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

tf_basic_model.submit_prediction(model=gradient_regressor,
                                 testing_examples=testing_examples,
                                 testing_targets=testing_targets)

adam_regressor, adam_training_losses, adam_validation_losses = tf_basic_model.train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=10000,
    batch_size=50,
    hidden_units=[2048, 512, 32],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

tf_basic_model.submit_prediction(model=adam_regressor,
                                 testing_examples=testing_examples,
                                 testing_targets=testing_targets,
                                 filename='adam_submission')

plt.ylabel("RMSE")
plt.xlabel("Periods")
plt.title("Root Mean Squared Error vs. Periods")
plt.plot(gradient_training_looses, label='Gradient training')
plt.plot(gradient_validation_losses, label='Gradient validation')
plt.plot(adam_training_losses, label='Adam training')
plt.plot(adam_validation_losses, label='Adam validation')
_ = plt.legend()
plt.show()
