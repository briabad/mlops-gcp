
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