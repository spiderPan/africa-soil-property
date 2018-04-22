import math
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)


class tf_basic_model:
    def preprocess_features(data_frame):
        preprocess_features = data_frame.copy()
        label_features_indexes = pd.Index(['Ca', 'P', 'pH', 'SOC', 'Sand'])
        preprocess_features_index = preprocess_features.columns.difference(label_features_indexes)
        selected_features = preprocess_features[preprocess_features_index]

        obj_cols = selected_features.select_dtypes(include=['object']).columns.drop('PIDN')
        selected_features[obj_cols] = selected_features[obj_cols].apply(lambda x: x.astype('category').cat.codes)

        return selected_features

    def preprocess_targets(data_frame):
        output_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']
        output_targets = pd.DataFrame()
        is_all_contains = True
        for output_col in output_cols:
            if not data_frame.columns.contains(output_col):
                is_all_contains = False
                break
        if is_all_contains:
            output_targets = data_frame[output_cols]
        else:
            output_targets = pd.DataFrame(0, index=np.arange(len(data_frame)), columns=output_cols)
        return output_targets

    def construct_feature_columns(input_features):
        return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features.drop(['PIDN'], axis=1)])

    def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
        features = {key: np.array(value) for key, value in dict(features).items()}

        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def train_nn_regression_model(
            my_optimizer,
            steps,
            batch_size,
            hidden_units,
            training_examples,
            training_targets,
            validation_examples,
            validation_targets):
        periods = 20
        steps_per_period = steps / periods

        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=tf_basic_model.construct_feature_columns(training_examples),
            hidden_units=hidden_units,
            optimizer=my_optimizer,
            label_dimension=5
        )
        target_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']

        def training_input_fn(): return tf_basic_model.my_input_fn(training_examples,
                                                                   training_targets[target_cols],
                                                                   batch_size=batch_size)

        def predict_training_input_fn(): return tf_basic_model.my_input_fn(training_examples,
                                                                           training_targets[target_cols],
                                                                           num_epochs=1,
                                                                           shuffle=False)

        def predict_validation_input_fn(): return tf_basic_model.my_input_fn(validation_examples,
                                                                             validation_targets[target_cols],
                                                                             num_epochs=1,
                                                                             shuffle=False)

        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        for period in range(0, periods):
            dnn_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period,
            )
            # Take a break and compute predictions.
            training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
            training_predictions = np.array([item['predictions'] for item in training_predictions])

            validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'] for item in validation_predictions])

            # Compute training and validation loss.
            training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" %
                  (period, training_root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
        print("Model training finished.")

        # Output a graph of loss metrics over periods.
        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.legend()

        print("Final RMSE (on training data):   %0.2f" %
              training_root_mean_squared_error)
        print("Final RMSE (on validation data): %0.2f" %
              validation_root_mean_squared_error)

        return dnn_regressor, training_rmse, validation_rmse

    def get_input_fn(data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(x=pd.DataFrame({k: data_set[k].values for k in data_set.columns}), y=None, num_epochs=num_epochs, shuffle=shuffle)

    def submit_prediction(model, testing_examples, testing_targets, filename=None):
        if filename is None:
            filename = 'submission'

        target_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']

        def predict_testing_input_fn(): return tf_basic_model.my_input_fn(testing_examples, testing_targets[target_cols], num_epochs=1, shuffle=False)
        predictions = model.predict(input_fn=predict_testing_input_fn)
        predictions = np.array([item['predictions'] for item in predictions])
        submission = pd.DataFrame(predictions, columns=target_cols)
        submission['PIDN'] = testing_examples['PIDN']
        submission = submission.reindex(columns=['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand'])
        submission.head()
        submission.to_csv('./data/submission.csv', index=False)
