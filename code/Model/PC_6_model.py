import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv1D, LSTM
from keras.layers import LayerNormalization, MultiHeadAttention
from keras.layers import Add
from keras import optimizers
import os
import numpy as np


def transformer_block(x, num_heads=4, ff_dim=128):
    # Multi-head self attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn_output = Add()([x, attn_output])          # Residual connection
    out1 = LayerNormalization()(attn_output)

    # Feed Forward network
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(x.shape[-1])(ffn)
    ffn_output = Add()([out1, ffn])                # Residual connection
    out2 = LayerNormalization()(ffn_output)

    return out2


def t_m(train_data, train_label, model_name, path=None):
    # Ensure a valid path to save models/logs
    if path is None:
        path = os.path.join(os.getcwd(), 'models')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # INPUT (200, 6)
    input_ = Input(shape=(200, 6))

    # -------------------------
    # 1. CNN feature extractor
    # -------------------------
    x = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(input_)
    x = Dropout(0.4)(x)
    x = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # -------------------------
    # 2. Transformer stack
    # -------------------------
    for _ in range(2):
        x = transformer_block(x, num_heads=4, ff_dim=256)
        x = Dropout(0.3)(x)

    # -------------------------
    # 3. LSTM decoder
    # -------------------------
    x = LSTM(units=100, return_sequences=False, dropout=0.3)(x)
    x = Dropout(0.4)(x)

    # -------------------------
    # 4. Output Layer
    # -------------------------
    output = Dense(1, activation='sigmoid')(x)

    # Build and compile model
    model = Model(inputs=input_, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    best_weights_filepath = os.path.join(path, f"{model_name}_best_weights.h5")
    saveBestModel = keras.callbacks.ModelCheckpoint(
        best_weights_filepath,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='auto'
    )

    CSVLogger = keras.callbacks.CSVLogger(
        os.path.join(path, f"{model_name}_csvLogger.csv"),
        separator=',',
        append=False
    )

    # Train
    batch_size = max(1, int(0.5 * len(train_data)))
    
    history = model.fit(
        train_data,
        train_label,
        shuffle=True,
        validation_split=0.1,
        epochs=200,
        batch_size=batch_size,
        callbacks=[saveBestModel, CSVLogger, early_stop]
    )

    return model
