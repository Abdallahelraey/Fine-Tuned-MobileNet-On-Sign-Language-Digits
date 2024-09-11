import unittest
import numpy as np
import tensorflow as tf
from src.models.model import SignLanguageModel

class TestSignLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model = SignLanguageModel(input_shape=(224, 224, 3), num_classes=10)
        self.model.compile_model()

    def test_model_initialization(self):
        self.assertIsInstance(self.model, SignLanguageModel)
        self.assertEqual(self.model.input_shape, (224, 224, 3))
        self.assertEqual(self.model.num_classes, 10)

    def test_model_architecture(self):
        self.assertIsInstance(self.model.model, tf.keras.Model)
        self.assertEqual(self.model.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(self.model.model.output_shape, (None, 10))

    def test_compile_model(self):
        self.assertIsNotNone(self.model.model.optimizer)
        self.assertEqual(self.model.model.loss, 'categorical_crossentropy')
        # metric_names = [m.name for m in self.model.model.metrics]
        # self.assertIn('accuracy', metric_names)

    def test_train(self):
        # Create dummy data for training
        X_train = np.random.rand(100, 224, 224, 3)
        y_train = np.random.randint(0, 10, size=(100, 10))
        X_val = np.random.rand(20, 224, 224, 3)
        y_val = np.random.randint(0, 10, size=(20, 10))

        history = self.model.train((X_train, y_train), (X_val, y_val), epochs=2, batch_size=32)
        self.assertIn('accuracy', history.history)
        self.assertIn('val_accuracy', history.history)

    def test_evaluate(self):
        X_test = np.random.rand(50, 224, 224, 3)
        y_test = np.random.randint(0, 10, size=(50, 10))

        test_loss, test_accuracy = self.model.evaluate((X_test, y_test))
        self.assertIsInstance(test_loss, float)
        self.assertIsInstance(test_accuracy, float)

    def test_predict(self):
        X_pred = np.random.rand(10, 224, 224, 3)
        predictions = self.model.predict(X_pred)
        self.assertEqual(predictions.shape, (10, 10))

    def test_save_and_load_model(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, 'test_model.keras')
            self.model.save_model(filepath)
            self.assertTrue(os.path.exists(filepath))

            loaded_model = SignLanguageModel.load_model(filepath)
            self.assertIsInstance(loaded_model, tf.keras.Model)
            self.assertEqual(loaded_model.input_shape, (None, 224, 224, 3))
            self.assertEqual(loaded_model.output_shape, (None, 10))

if __name__ == '__main__':
    unittest.main()
