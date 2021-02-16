/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 * */

import * as tf from "@tensorflow/tfjs-node";
import {
	BLOCKS,
	DATA_FORMAT
}              from "./constants.js";

class Densenet extends tf.LayersModel {
	static className = 'Functional';
	
	constructor(args) {
		super(args)
	}
}

/**
 * A building block for a dense block.
 *
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} growth_rate: float, growth rate at dense layers.
 * @param {string} name block label.
 * @param {boolean} debug.
 *
 * @returns {Tensor}|{TensorLike} Output tensor for the block.
 */
function convBlock(name, inputs, growth_rate,
                   debug = false) {
	const axis = (DATA_FORMAT === 'channelsLast') ? 3 : 1;
	
	let outputs = tf.layers.batchNormalization({axis: axis, epsilon: 1.001e-5, name: name + '_0_bn'})
		.apply(inputs);
	
	outputs = tf.layers.activation({name: name + '_0_relu', activation: 'relu', dtype: 'float32'})
		.apply(outputs);
	
	outputs = tf.layers.conv2d({
		                           name:  name + '_1_conv', filters: 4 * growth_rate, kernelSize: 1, useBias: false,
		                           dtype: 'float32', kernelInitializer: 'glorotUniform'
	                           })
		.apply(outputs);
	
	outputs = tf.layers.batchNormalization({name: name + '_1_bn', axis: [axis], epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({name: name + '_1_relu', activation: 'relu'})
		.apply(outputs);
	
	outputs = tf.layers.conv2d(
		{
			name:  name + '_2_conv', filters: growth_rate, kernelSize: 3, padding: 'same', useBias: false,
			dtype: 'float32', kernelInitializer: 'glorotUniform'
		}
	).apply(outputs);
	
	inputs = tf.layers.concatenate(
		{name: name + '_concat', axis: axis, dtype: 'float32'}
	).apply([inputs, outputs]);
	
	if (debug)
		console.debug("conv_block", inputs.shape)
	
	return inputs;
}

/**
 * A dense block.
 *
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} blocks: the number of building blocks.
 * @param {string} name: block label.
 * @param {boolean} debug.
 * @returns {Tensor} Output tensor for the block.
 */
function denseBlock(name, inputs, blocks,
                    debug = false) {
	for (let i = 0; i < blocks; ++i)
		inputs = convBlock(name + '_block' + (i + 1).toString(), inputs, 32, debug);
	return inputs;
}

/**
 * A transition block.
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} reduction compression rate at transition layers.
 * @param {string} name block label.
 * @param {boolean} debug.
 *
 * @returns {Tensor} output tensor for the block.
 * */
function transitionBlock(name, inputs, reduction,
                         debug = false) {
	const axis = (DATA_FORMAT === 'channelsLast') ? 3 : 1;
	inputs = tf.layers.batchNormalization(
		{
			name:    name + '_bn',
			axis:    [axis],
			dtype:   'float32',
			epsilon: 1.001e-5
		}
	).apply(inputs);
	
	inputs = tf.layers.activation(
		{
			name:       name + '_relu',
			activation: 'relu',
			dtype:      'float32',
		}
	).apply(inputs);
	
	inputs = tf.layers.conv2d(
		{
			name:    name + '_conv', filters: parseInt(inputs.shape[axis] * reduction),
			dtype:   'float32', dilationRate: 1, kernelInitializer: 'glorotUniform',
			strides: 1, kernelSize: 1, useBias: false
		}
	).apply(inputs);
	
	let outputs = tf.layers.averagePooling2d(
		{
			name:    name + '_pool', poolSize: 2,
			strides: 2, dtype: 'float32',
		}
	).apply(inputs);
	
	if (debug)
		console.debug("transitionBlock:", outputs.shape)
	
	return outputs;
}

function densenetPrepare(inputs, debug = false) {
	const axis = (DATA_FORMAT === 'channelsLast') ? 3 : 1;
	let outputs = tf.layers.zeroPadding2d({name: "zero_padding2d", padding: [[3, 3], [3, 3]]})
		.apply(inputs);
	
	outputs = tf.layers.conv2d({
		                           name:              'conv1/conv', filters: 64, kernelSize: 7,
		                           kernelInitializer: 'glorotUniform', strides: 2,
		                           activation:        'linear', padding: 'valid', useBias: false
	                           })
		.apply(outputs);
	
	outputs = tf.layers.batchNormalization({name: 'conv1/bn', axis: axis, epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({name: 'conv1/relu', activation: 'relu', dtype: 'float32'})
		.apply(outputs);
	
	outputs = tf.layers.zeroPadding2d({
		                                  name:  'zero_padding2d_1', padding: [[1, 1], [1, 1]],
		                                  dtype: 'float32'
	                                  })
		.apply(outputs);
	
	outputs = tf.layers.maxPooling2d({name: 'pool1', poolSize: 3, strides: 2, dtype: 'float32'})
		.apply(outputs);
	
	outputs = denseBlock('conv2', outputs, BLOCKS[0], debug);
	outputs = transitionBlock('pool2', outputs, 0.5, debug);
	outputs = denseBlock('conv3', outputs, BLOCKS[1], debug);
	outputs = transitionBlock('pool3', outputs, 0.5, debug);
	outputs = denseBlock('conv4', outputs, BLOCKS[2], debug);
	outputs = transitionBlock('pool4', outputs, 0.5, debug);
	outputs = denseBlock('conv5', outputs, BLOCKS[3], debug);
	
	outputs = tf.layers.batchNormalization({name: 'bn', axis: [axis], epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({name: 'relu', activation: 'relu'})
		.apply(outputs);
	
	return outputs
}

export function densenet(shape, inputs = null, debug = false) {
	
	if (inputs === null)
		inputs = tf.input({shape: shape});
	
	const outputs = densenetPrepare(inputs, debug)
	
	return new Densenet({name: 'densenet121', inputs: inputs, outputs: outputs});
}
