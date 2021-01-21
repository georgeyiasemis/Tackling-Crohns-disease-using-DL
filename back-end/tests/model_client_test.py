"""
Test to check that the model_client.py outputs the same values as Robert
Holland (https://github.com/RobbieHolland/CrohnsDisease) inhereted code.
Note: the tensorflow-servering server must run with the models present in the
"/CrohnsDisease/inhereted_trained_models" and with the specified coordiantes to pass.
"""

import unittest
import numpy as np

from model_client import query_client, pre_process_image, get_prediction
from predict_client.prod_client import ProdClient


coords = [130, 150, 26] # the test work only for these coordinates
record_shape = [37,99,99]
feature_shape = [31,87,87]


class ModelClientTest(unittest.TestCase):

    def test_pre_process_image(self):
        # get acutal image from running Robbie's code
        actual_image = np.load("./tests/actual_model_input_image.npy")
        actual_image = actual_image.reshape([-1, 1] + feature_shape)

        # get Image from running code written
        testing_image = pre_process_image(coords, "../examples/A1 Axial T2.nii")
        # check if equal
        np.testing.assert_array_equal(testing_image, actual_image)


    def test_query_client(self):
        # get result from running Robbie's code
        actual_probs = np.load("./tests/actual_output_probabilities.npy")

        # get result from running our code
        image = np.load("./tests/actual_model_input_image.npy")
        image = image.reshape([-1, 1] + feature_shape)

        client = ProdClient("dudley.doc.ic.ac.uk"+':9200', 'crohns', 1)

        testing_probs, _ = query_client(image, client)
        # check if equal
        np.testing.assert_array_equal(actual_probs, testing_probs)

    def test_get_prediction(self):

        # Actual output from txt gotten after running robbie's code
        with open('./actual_output_string.txt', 'r') as f:
            actual_string = f.readlines()[-2]

        # string from running get predictions
        test_string = get_prediction(coords, "../examples/A1 Axial T2.nii", "dudley.doc.ic.ac.uk")

        # check if equal
        self.assertEqual(actual_string[:-1], test_string)



if __name__ == "__main__":

    unittest.main()
