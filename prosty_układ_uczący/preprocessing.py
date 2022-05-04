import os
import librosa
import math
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

def plots(history):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(history.history["accuracy"], label="train accuracy", color="lightcoral")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy", color="darkred")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Plot")
    
    axs[1].plot(history.history["loss"], label="train error", color="slateblue")
    axs[1].plot(history.history["val_loss"], label="test error",color="darkblue" )
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Error Plot")

    plt.show()


PATH_TO_DATASET = "/mnt/c/Users/niczk/OneDrive/Documents/studies/4sem/biologiczne/projekt1/MusicGenreClassification/archive/Data/genres_original"
PATH_TO_JSON = "/mnt/c/Users/niczk/OneDrive/Documents/studies/4sem/biologiczne/projekt1/MusicGenreClassification/json_files/data.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(path_to_dataset, path_to_json, mfcc_number=13, fft_number = 2048, hop_length=512, segments_number=10):
    data_dict = {
        "genres": [],
        "mfcc" : [],
        "labels": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / segments_number)
    expected_mfcc_segment_no = math.ceil(samples_per_segment / hop_length)
    for i, (curr_dir_path, curr_dir_names, curr_files) in enumerate(os.walk(path_to_dataset)):
        if curr_dir_path is not path_to_dataset:
            data_dict["genres"].append(curr_dir_path.split("/")[-1])
            print("\nProcessing {}".format(curr_dir_path.split("/")[-1]))

            for f in curr_files:
                actual_path = os.path.join(curr_dir_path, f)
                #sr, signal = sio.wavfile.read(actual_path)
                print(actual_path)
                signal, sr = librosa.load(actual_path, sr=SAMPLE_RATE)

                for s in range(segments_number):
                    sample_start = samples_per_segment * s
                    sample_end = sample_start + samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[sample_start:sample_end], sr= sr, n_fft = fft_number, n_mfcc = mfcc_number, hop_length = hop_length, )
                    mfcc = mfcc.T
                    if len(mfcc) == expected_mfcc_segment_no:
                        data_dict["mfcc"].append(mfcc.tolist())
                        data_dict["labels"].append(i-1)

    with open(path_to_json, "w", encoding='utf-8') as fp:
        json.dump(data_dict, fp, indent=4)


def load_data(json_path):
    with open(json_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


if __name__ == '__main__':
    #save_mfcc(PATH_TO_DATASET, PATH_TO_JSON)
    inputs, targets = load_data(PATH_TO_JSON)
    inputs_training, inputs_test, targets_training, targets_test = train_test_split(inputs, targets, test_size=0.3)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

    #, kernel_regularizer=keras.regularizers.l2(0.001)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(inputs_training, targets_training, validation_data=(inputs_test, targets_test), epochs=100, batch_size=32)

    plots(history)


