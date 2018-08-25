from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import numpy.testing as npt  # type: ignore
import numpy.random as npr

from PIL import Image  # type: ignore

from onnx_coreml import convert
from tests._test_utils import _onnx_create_single_node_model, _onnx_create_model
from onnx import helper, TensorProto


class ConvertTest(unittest.TestCase):
    def setUp(self):  # type: () -> None
        self.img_arr = np.uint8(npr.rand(224, 224, 3) * 255)  # type: ignore
        self.img = Image.fromarray(np.uint8(self.img_arr))  # type: ignore
        self.img_arr = np.float32(self.img_arr)  # type: ignore
        self.onnx_model = _onnx_create_single_node_model(
            "Relu",
            [(3, 224, 224)],
            [(3, 224, 224)]
        )
        self.input_names = [i.name for i in self.onnx_model.graph.input]
        self.output_names = [o.name for o in self.onnx_model.graph.output]

    def test_convert_image_input(self):  # type: () -> None
        coreml_model = convert(
            self.onnx_model,
            image_input_names=self.input_names
        )
        spec = coreml_model.get_spec()
        for input_ in spec.description.input:
            self.assertEqual(input_.type.WhichOneof('Type'), 'imageType')

    def test_convert_image_output(self):  # type: () -> None
        coreml_model = convert(
            self.onnx_model,
            image_output_names=self.output_names
        )
        spec = coreml_model.get_spec()
        for output in spec.description.output:
            self.assertEqual(output.type.WhichOneof('Type'), 'imageType')

    def test_convert_image_input_preprocess(self):  # type: () -> None
        bias = np.array([100, 90, 80])
        coreml_model = convert(
            self.onnx_model,
            image_input_names=self.input_names,
            preprocessing_args={
                'is_bgr': True,
                'blue_bias': bias[0],
                'green_bias': bias[1],
                'red_bias': bias[2]
            }
        )
        output = coreml_model.predict(
            {
                self.input_names[0]: self.img
            }
        )[self.output_names[0]]

        expected_output = self.img_arr[:, :, ::-1].transpose((2, 0, 1))
        expected_output[0] = expected_output[0] + bias[0]
        expected_output[1] = expected_output[1] + bias[1]
        expected_output[2] = expected_output[2] + bias[2]
        npt.assert_equal(output, expected_output)

    def test_convert_image_output_bgr(self):  # type: () -> None
        coreml_model = convert(
            self.onnx_model,
            image_input_names=self.input_names,
            image_output_names=self.output_names,
            deprocessing_args={
                'is_bgr': True
            }
        )
        output = coreml_model.predict(
            {
                self.input_names[0]: self.img
            }
        )[self.output_names[0]]
        output = np.array(output)[:, :, :3].transpose((2, 0, 1))
        expected_output = self.img_arr[:, :, ::-1].transpose((2, 0, 1))
        npt.assert_equal(output, expected_output)

    def test_image_scaler_remover(self): # type: () -> None
        inputs = [('input', (1,3,50,50))]
        outputs = [('out', (1,3,50,50), TensorProto.FLOAT)]

        im_scaler = helper.make_node("ImageScaler",
                                     inputs = ['input'],
                                     outputs = ['scaler_out'],
                                     bias = [10,-6,20], scale=3.0)

        exp = helper.make_node("Exp",
                               inputs=["scaler_out"],
                               outputs=['out'])

        onnx_model = _onnx_create_model([im_scaler, exp], inputs, outputs)

        spec = convert(onnx_model).get_spec()
        self.assertEqual(spec.description.input[0].name, 'input')
        self.assertEqual(spec.description.output[0].name, 'out')
        self.assertEqual(len(spec.neuralNetwork.layers), 1)
        self.assertEqual(len(spec.neuralNetwork.preprocessing), 1)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.channelScale, 3.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.blueBias, 20.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.greenBias, -6.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.redBias, 10.0)

    def test_multiple_image_scaler(self): # type : () -> None
        inputs = [('input_color', (1,3,10,10)), ('input_gray', (1,1,10,10))]
        outputs = [('out', (1,4,10,10), TensorProto.FLOAT)]

        im_scaler1 = helper.make_node("ImageScaler",
                                     inputs = ['input_color'],
                                     outputs = ['scaler_out_1'],
                                     bias = [10,-6,20], scale=3.0)

        im_scaler2 = helper.make_node("ImageScaler",
                                     inputs = ['input_gray'],
                                     outputs = ['scaler_out_2'],
                                     bias = [-13], scale=5.0)

        concat = helper.make_node("Concat",
                                  inputs=['scaler_out_1', 'scaler_out_2'],
                                  outputs=['out'],
                                  axis = 1)

        onnx_model = _onnx_create_model([im_scaler1, im_scaler2, concat], inputs, outputs)

        spec = convert(onnx_model).get_spec()
        self.assertEqual(len(spec.neuralNetwork.layers), 1)
        self.assertEqual(len(spec.neuralNetwork.preprocessing), 2)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.channelScale, 3.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.blueBias, 20.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.greenBias, -6.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[0].scaler.redBias, 10.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[1].scaler.channelScale, 5.0)
        self.assertEqual(spec.neuralNetwork.preprocessing[1].scaler.grayBias, -13.0)


if __name__ == '__main__':
    unittest.main()
