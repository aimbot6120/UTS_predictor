from tensorflow.keras.layers import Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.regularizers import l2

def build_model():
    # Chemical composition input
    chemical_input = Input(shape=(23,), name='chemical_input')
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(chemical_input)
    x1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x1)
    chemical_output = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x1)

    # Image input
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-4]:  # Set the last 4 layers to not trainable
        layer.trainable = False
    x2 = base_model(image_input)
    x2 = Flatten()(x2)
    x2 = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x2)
    x2 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x2)
    image_output = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x2)

    # Concatenate the outputs of the two models
    combined = concatenate([chemical_output, image_output], axis=-1)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(image_output)
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)

    output_layer = Dense(1, activation='linear', name='output')(combined)

    model = Model(inputs=[chemical_input, image_input], outputs=[output_layer])

    return model
