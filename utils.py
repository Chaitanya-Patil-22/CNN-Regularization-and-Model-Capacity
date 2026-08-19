import numpy as np
import h5py



def load_dataset():
    train_dataset=h5py.File(r'data_set/train_signs.h5',"r")
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])
    train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))

    test_dataset=h5py.File(r'data_set/test_signs.h5',"r")
    test_set_x_orig=np.array(test_dataset["test_set_x"][:])
    test_set_y_orig=np.array(test_dataset["test_set_y"][:])
    test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig

def one_hot(Y):
    Y=Y.reshape(-1)
    A=np.eye(6)[Y]

    return A

    





