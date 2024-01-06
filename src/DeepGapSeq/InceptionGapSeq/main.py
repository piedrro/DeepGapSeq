import pickle
import os
import importlib
import importlib.resources as resources
from DeepGapSeq.InceptionGapSeq.dataloader import create_dataset, read_gapseq_data
from DeepGapSeq.InceptionGapSeq.architecture_tf import build_model
import tensorflow as tf
from glob2 import glob
import keras
from sklearn.model_selection import train_test_split
import numpy as np


ratio_train = 0.7
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 5
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "TEST"

directory_path = r"F:/Traces for ML_GAP-sequencing"
output_directory = './models/'

complimentary_files_path = os.path.join(directory_path, "Complementary Traces")
noncomplimentary_files_path = os.path.join(directory_path, "Non_Complementary Traces")

complimentary_files = glob(complimentary_files_path + "*/*_gapseqML.txt")
noncomplimentary_files = glob(noncomplimentary_files_path + "*/*_gapseqML.txt")

X, y, file_names = read_gapseq_data(complimentary_files, label=0, trace_limit=1200)
X, y, file_names = read_gapseq_data(noncomplimentary_files, X, y, file_names, label=1, trace_limit=1200)

if __name__ == '__main__':
   
   
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      train_size=ratio_train,
                                                      random_state=42,
                                                      shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                    train_size=val_test_split,
                                                    random_state=42,
                                                    shuffle=True)
    
    training_dataset = create_dataset(data = X_train,
                                    labels = y_train,
                                    augment = True)
    
    validation_dataset = create_dataset(data = X_val,
                                    labels = y_val,
                                      augment=False)
    
    test_dataset = create_dataset(data = X_test,
                                labels = y_test,
                                augment=False)
    
    model = build_model(1,len(np.unique(y)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=output_directory, monitor='loss',
                                                        save_best_only=True)
    earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    callbacks = [reduce_lr, model_checkpoint, earlystopping]
   
    hist = model.fit(training_dataset, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        verbose=True, 
                        validation_data=validation_dataset,
                        callbacks=callbacks)

    model.save(os.path.join(output_directory, 'model.h5'))