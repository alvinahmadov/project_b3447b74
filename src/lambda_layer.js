/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 * */
import * as tf from "@tensorflow/tfjs-node";

export class Lambda extends tf.layers.Layer {
	static className = 'Lambda'
	
	constructor(func, kwargs = {}) {
		if (kwargs.name === undefined)
			kwargs.name = 'lambda';
		super(kwargs);
		this.function = func;
		this.functionType = 'lambda';
	}
	
	/**
	 * This layer only works on 4D Tensors [batch, height, width, channels],
	 * and produces output with twice as many channels.
	 *
	 * layer.computeOutputShapes must be overridden in the case that the output
	 * shape is not the same as the input shape.
	 * @param {Array} inputShape
	 */
	computeOutputShape(inputShape) {
		return [inputShape[0], inputShape[2], inputShape[3]];
	}
	
	
	/**
	 * Centers the input and applies the following function to every element of
	 * the input.
	 *
	 *     x => [func(x)]
	 *
	 * @param inputs Tensor to be treated.
	 * @param kwargs Only used as a pass through to call hooks
	 */
	call(inputs, kwargs) {
		let input = inputs;
		if (Array.isArray(input))
			input = input[0];
		
		this.invokeCallHook(input, kwargs);
		return this.function(input, kwargs);
	}
}

tf.serialization.registerClass(Lambda);
