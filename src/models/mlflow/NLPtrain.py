from tkinter import Y
import warnings

import pandas as pd
import numpy as np

import mlflow
import mlflow.tensorflow
import mlflow.keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

import os
import warnings
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


from dotenv import find_dotenv, load_dotenv


def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results



if __name__ == "__main__":
  mlflow.keras.autolog()
  warnings.filterwarnings('ignore')
  np.random.seed(40)

  load_dotenv()
  train_df = pd.read_csv(os.getenv("TRAIN_DATA"))
  test_df = pd.read_csv(os.getenv("TEST_DATA"))

  train_df_shuffled = train_df.sample(frac=1, random_state=42) 
  train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, # dedicate 10% of samples to validation set
                                                                            random_state=42)

  EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
  CONV_UNITS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
  NUM_HIDDEN = int(sys.argv[3]) if len(sys.argv) > 3 else 1


  tf.random.set_seed(42)
  max_vocab_length = 10000 # max number of words to have in our vocabulary
  max_length = 15 # max length our sequences will be (how many words from a Tweet does our model see?)

  text_vectorizer = TextVectorization(max_tokens=max_vocab_length,# how many words in the vocabulary (all of the different words in your text)
                                      output_mode="int",# how to map tokens to numbers
                                      output_sequence_length=max_length)#

  text_vectorizer.adapt(train_sentences)
  model_3_embedding = layers.Embedding(input_dim=max_vocab_length,
                                      output_dim=128,
                                      embeddings_initializer="uniform",
                                      input_length=max_length,
                                      name="embedding_3")

 

  # Create 1-dimensional convolutional layer to model sequences
  inputs = layers.Input(shape=(1,), dtype="string")
  x = text_vectorizer(inputs)
  x = model_3_embedding(x)

  for size in range(NUM_HIDDEN): 
    x = layers.Conv1D(filters=CONV_UNITS, kernel_size=5, activation="relu", padding='same',input_shape=(None,CONV_UNITS))(x)
    x = layers.Conv1D(filters=CONV_UNITS, kernel_size=5, activation="relu", padding='same',input_shape=(None,CONV_UNITS))(x)
    
  x = layers.GlobalMaxPool1D()(x)
  #x = layers.GlobalMaxPool1D()(x)
  # x = layers.Dense(64, activation="relu")(x) # optional dense layer
  outputs = layers.Dense(1, activation="sigmoid")(x)
  CNN_NLP = tf.keras.Model(inputs, outputs, name="model_3_Conv1D")

  # Compile Conv1D model
  CNN_NLP.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

  # Fit the model
  CNN_NLP_history = CNN_NLP.fit(train_sentences,
                              train_labels,
                              epochs=EPOCHS,
                              validation_data=(val_sentences, val_labels),
                              )

  CNN_NLP.save_weights('Weights\CNN_NLP_weigths')
  model_pred_probs = CNN_NLP.predict(val_sentences) 
  # Convert model_5 prediction probabilities to labels
  model_preds = tf.squeeze(tf.round(model_pred_probs))
  model_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_preds)


  mlflow.log_param("Epochs", EPOCHS)
  mlflow.log_param("Units en capa CNN", CONV_UNITS)
  mlflow.log_param("Numero de capas escondidas", NUM_HIDDEN)
  mlflow.log_metric("Accuracy", model_results["accuracy"])
  mlflow.log_metric("Precision", model_results["precision"])
  mlflow.log_metric("F1", model_results["f1"])
  mlflow.log_metric("Recall", model_results["recall"])

  mlflow.log_artifacts(r"C:\Users\w10\Desktop\NLP\models\MlFlow\Weights")
  mlflow.keras.log_model(CNN_NLP, "model")
