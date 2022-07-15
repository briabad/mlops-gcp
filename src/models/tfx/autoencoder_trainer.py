from typing import Dict, List, Text
import tensorflow as tf 
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx 
from tfx_bsl.public import tfxio
from tensorflow_transform import TFTransformOutput
import autoencoder_constant


NUMERICAL_FEATURES = autoencoder_constant.NUMERICAL_FEATURES

_FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
           for feature in NUMERICAL_FEATURES
       },
**{feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for feature in NUMERICAL_FEATURES },
}

def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
    A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=NUMERICAL_FEATURES),
        tf_transform_output.transformed_metadata.schema)



class Autoencoder(keras.Model):
  """ Deep learning model. This object let you create an autoencoder, you specify the number of units neurons and the number of hidden layers, for the decoder and the encoder, 
  the same number of hidden layers and units neurons is taken to create both. also you specify the mid layer and the input shape  

  Args:
      keras (object): Model inheritance
  """

  def __init__(self,input_shape,intermediary_layers,z_layer,name='Autoencoder'):
    """function wich initialize the object

    Args:
        input_shape (int): number of the input shape of the model
        intermediary_layers (List): number of units nuerons and hidden layers 
        z_layer (int): mid layer
        name (str, optional): description of the model. Defaults to 'Autoencoder'.
    """
    tf.random.set_seed(3)
    super(Autoencoder,self).__init__(name=name,)
    self.z_layer = z_layer
    self.intermediary_layers = intermediary_layers
    self.encoder = keras.models.Sequential([
                                            keras.layers.InputLayer(input_shape = input_shape),
                                            self.dense_layers(self.intermediary_layers),
                                            keras.layers.Dense(units = self.z_layer, activation = 'relu')

    ])
    self.decoder = keras.models.Sequential([
                                            self.dense_layers(reversed(self.intermediary_layers)),
                                            keras.layers.Dropout(0.2),
                                            keras.layers.Dense(units = input_shape,activation='sigmoid')
    ])

  def dense_layers(self,sizes):
    """ create the hidden layers

    Args:
        sizes (List): number of units nuerons and hidden layers 

    Returns:
        object: keras model wich contain the hidden layers
    """
    model = keras.Sequential()
    
    for size in sizes: 
      model.add( tf.keras.layers.Dense(size, activation='relu'))
      model.add(keras.layers.Dropout(0.1))
    
    return model                   

  def call(self,x):
    """ method wich let the call of the model

    Args:
        x (DataFrame, Array): input

    Returns:
        object: Model 
    """
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    
    return decoded

def build_keras_model():
    model = Autoencoder(len(NUMERICAL_FEATURES),intermediary_layers=[300,200,100,50,25],z_layer=30)

    model.compile(optimizer='adam',metrics=[keras.metrics.mean_squared_error], loss=keras.losses.MeanSquaredError())

    return model


def keras_train_and_evaluate(model, train_dataset, validation_dataset, epochs=10):

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,restore_best_weights=True)
    history = model.fit(x=train_dataset, y=train_dataset,
                    epochs=100,
                    shuffle=True,
                    validation_data=(validation_dataset, validation_dataset),callbacks = [early_stopping] )
    #Add callbacks

    return model

# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # This schema is usually either an output of SchemaGen or a manually-curated
  # version provided by pipeline author. A schema can also derived from TFT
  # graph if a Transform component is used. In the case when either is missing,
  # `schema_from_feature_spec` could be used to generate schema from very simple
  # feature_spec, but the schema returned would be very primitive.
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=20)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=10)

  model = build_keras_model()
  model = keras_train_and_evaluate(model, train_dataset, eval_dataset, fn_args.custom_config['epochs'])

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')
