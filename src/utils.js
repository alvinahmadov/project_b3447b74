import jsYaml  from "js-yaml";
import {JSDOM} from "jsdom";

import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import {cv}    from "opencv-wasm";
import canvas  from "canvas"

const {loadImage, createCanvas, Canvas, Image, ImageData} = canvas;

const BLOCKS = [6, 12, 24, 16];
const AXIS = 3;
const DEBUG = false;

function cvTranslateError(err) {
	let error_stmt = undefined;
	
	if (typeof err === 'undefined') {
		error_stmt = '';
	} else if (typeof err === 'number') {
		if (!isNaN(err)) {
			if (typeof cv !== 'undefined') {
				// noinspection JSUnresolvedFunction
				error_stmt = 'Exception: ' + cv.exceptionFromPtr(err).msg;
			}
		}
	} else if (typeof err === 'string') {
		let ptr = Number(err.split(' ')[0]);
		if (!isNaN(ptr)) {
			if (typeof cv !== 'undefined') {
				// noinspection JSUnresolvedFunction
				error_stmt = 'Exception: ' + cv.exceptionFromPtr(ptr).msg;
			}
		}
	} else if (err instanceof Error) {
		error_stmt = err;
	}
	
	return error_stmt;
}

function assert(expr, msg) {
	if (!expr) {
		throw new Error(typeof msg === 'string' ? msg : msg());
	}
}

function toList(x) {
	if (Array.isArray(x)) {
		return x;
	}
	return [x];
}

/**
 * @param {number|number[]} axis
 * @param {number[]} shape
 * @returns {number[]}
 * */
export function parseAxisParam(axis, shape) {
	const rank = shape.length;
	
	// Normalize input
	axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
	
	// Check for valid range
	assert(
		axis.every(ax => ax >= -rank && ax < rank),
		() =>
			`All values in axis param must be in range [-${rank}, ${rank}) but ` +
			`got axis ${axis}`);
	
	// Check for only integers
	assert(
		axis.every(ax => ax % 1 === 0),
		() => `All values in axis param must be integers but ` +
			`got axis ${axis}`);
	
	// Handle negative axis.
	return axis.map(a => a < 0 ? rank + a : a);
}

/**
 * Reduces the shape by removing all dimensions of shape 1.
 * @param {number[]} shape Shape to reshape
 * @param {number[]|number} axis
 * */
export function squeezeShape(shape, axis = null) {
	const newShape = [];
	const keptDims = [];
	const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
	const axes = (axis == null || isEmptyArray) ?
	             null :
	             parseAxisParam(axis, shape).sort();
	let j = 0;
	for (let i = 0; i < shape.length; ++i) {
		if (axes != null) {
			if (axes[j] === i && shape[i] !== 1) {
				throw new Error(
					`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
			}
			if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
				newShape.push(shape[i]);
				keptDims.push(i);
			}
			if (axes[j] <= i) {
				j++;
			}
		}
		if (shape[i] !== 1) {
			newShape.push(shape[i]);
			keptDims.push(i);
		}
	}
	return {newShape, keptDims};
}

/**
 * A building block for a dense block.
 *
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} growth_rate: float, growth rate at dense layers.
 * @param {string} name block label.
 *
 * @returns {Tensor}|{TensorLike} Output tensor for the block.
 */
function convBlock(name, inputs, growth_rate) {
	let outputs = tf.layers.batchNormalization({axis: AXIS, epsilon: 1.001e-5, name: name + '_0_bn'})
		.apply(inputs);
	
	outputs = tf.layers.activation({ name: name + '_0_relu', activation: 'relu'})
		.apply(outputs);
	
	outputs = tf.layers.conv2d({name: name + '_1_conv', filters: 4 * growth_rate, kernelSize: 1, useBias: false})
		.apply(outputs);
	
	outputs = tf.layers.batchNormalization({name: name + '_1_bn' ,axis: AXIS, epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({ name: name + '_1_relu', activation: 'relu'})
		.apply(outputs);
	
	outputs = tf.layers.conv2d(
		{name: name + '_2_conv', filters: growth_rate, kernelSize: 3, padding: 'same', useBias: false}
		).apply(outputs);
	
	inputs = tf.layers.concatenate(
		{name: name + '_concat', axis: AXIS}
	).apply([inputs, outputs]);
	
	if (DEBUG)
		console.log("conv_block", inputs.shape)
	
	return inputs;
}

/**
 * A dense block.
 *
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} blocks: the number of building blocks.
 * @param {string} name: block label.
 * @returns {Tensor} Output tensor for the block.
 */
function denseBlock(name, inputs, blocks) {
	for (let i = 0; i < blocks; ++i)
		inputs = convBlock(name + '_block' + (i + 1).toString(), inputs, 32);
	return inputs;
}

/**
 * A transition block.
 * Arguments:
 * @param {Tensor} inputs input tensor.
 * @param {number} reduction compression rate at transition layers.
 * @param {string} name block label.
 *
 * @returns {Tensor} output tensor for the block.
 * */
function transitionBlock(name, inputs, reduction) {
	inputs = tf.layers.batchNormalization(
		{
			name:    name + '_bn',
			axis:    AXIS,
			epsilon: 1.001e-5
		}
	).apply(inputs);
	
	inputs = tf.layers.activation(
		{
			name:       name + '_relu',
			activation: 'relu'
		}
	).apply(inputs);
	
	inputs = tf.layers.conv2d(
		{
			name:       name + '_conv',
			filters:    parseInt(inputs.shape[AXIS] * reduction),
			kernelSize: 1,
			useBias:    false
		}
	).apply(inputs);
	
	let outputs = tf.layers.averagePooling2d(
		{name:     name + '_pool', poolSize: 2,
			strides:  2
		}
	).apply(inputs);
	
	if (DEBUG)
		console.log("transitionBlock:", outputs.shape)
	
	return outputs;
}

export function densenet(input_shape) {
	let inputs = tf.layers.input({shape: input_shape});
	
	let outputs = tf.layers.zeroPadding2d({padding: [[3, 3], [3, 3]]})
		.apply(inputs);
	
	outputs = tf.layers.conv2d({name: 'conv1/conv', filters: 64, kernelSize: 7, strides: 2, useBias: false})
		.apply(outputs);
	
	outputs = tf.layers.batchNormalization({name: 'conv1/bn', axis: AXIS, epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({name: 'conv1/relu', activation: 'relu'})
		.apply(outputs);
	
	outputs = tf.layers.zeroPadding2d({padding: [[1, 1], [1, 1]]})
		.apply(outputs);
	
	outputs = tf.layers.maxPooling2d({name: 'pool1', poolSize: 3, strides: 2})
		.apply(outputs);
	
	outputs = denseBlock('conv2', outputs, BLOCKS[0]);
	outputs = transitionBlock('pool2', outputs, 0.5);
	outputs = denseBlock('conv3', outputs, BLOCKS[1]);
	outputs = transitionBlock('pool3', outputs, 0.5);
	outputs = denseBlock('conv4', outputs, BLOCKS[2]);
	outputs = transitionBlock('pool4', outputs, 0.5);
	outputs = denseBlock('conv5', outputs, BLOCKS[3]);
	
	outputs = tf.layers.batchNormalization({name: 'bn', axis: AXIS, epsilon: 1.001e-5})
		.apply(outputs);
	
	outputs = tf.layers.activation({name: 'relu', activation: 'relu'})
		.apply(outputs);
	
	return tf.model({name: 'densenet121', inputs: inputs, outputs: outputs});
}

export function cvToTensor(image) {
	let tensor = tf.tensor3d(image.data, [image.rows, image.cols, 3]);
	image.delete();
	tensor = tf.expandDims(tensor, 0);
	return tensor;
}

export function readParams(file_path, base = 'models') {
	const contents = fs.readFileSync(file_path)
	if (base)
		return jsYaml.load(contents)[base]
	else
		return jsYaml.load(contents)
}

export function getLetters(type_name) {
	const params = get_params('params.yaml')
	let letters = params[type_name]['letters']
	letters = letters.replace('$$', '$')
	return letters
}

export function saveModelAsJSON(path, model) {
	fs.writeFileSync(path, JSON.stringify(model))
}

export async function imread(img_path) {
	installDOM();
	let image = await loadImage(img_path);
	return cv.imread(image);
}

export async function imwrite(img, file_path) {
	try {
		const canvas = createCanvas(300, 300);
		cv.imshow(canvas, img);
		fs.writeFileSync(file_path, canvas.toBuffer('image/jpeg'));
	} catch (e) {
		console.log(cvTranslateError(e));
	}
}

function createMatVector(...args) {
	let matVector = new cv.MatVector();
	args.forEach(arg => matVector.push_back(arg));
	return matVector;
}

function installDOM() {
	const dom = new JSDOM();
	// noinspection JSConstantReassignment
	global.document = dom.window.document;
	global.Image = Image;
	global.HTMLCanvasElement = Canvas;
	global.ImageData = ImageData;
	global.HTMLImageElement = Image;
}

/**
 * @param {string} img_path Path to the image to be processed
 * @param {Object} params
 * @param {boolean} write Write the result image to filesystem
 *
 * @returns {cv.Mat}
 * */
export async function processImage(img_path,
                                    params,
                                    write = true) {
	const width = params['width'];
	const height = params['height'];
	const net_chanels = params['net_chanels'];
	const ratio = Math.floor(width / height);
	const color = params['fill_color'];
	try {
		let src = await imread(img_path);
		let dst = new cv.Mat();
		// Turn source image to RGB reducing the number of channels
		cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0);
		let w = src.cols;
		let h = src.rows;
		
		// Find image orientation and choose right op
		if (w / h !== ratio) {
			if (w / h < ratio) {
				w = (ratio * h) - w;
				let white = new cv.Mat.zeros(h, w, cv.CV_8U);
				// match the channel numbers of the two image
				cv.cvtColor(white, white, cv.COLOR_GRAY2BGR);
				white.data.fill(color);
				let matVector = createMatVector(src, white);
				// extend source image horizontally
				cv.hconcat(matVector, dst);
				white.delete();
			} else if (w / h > ratio) {
				let _h = Math.floor((w - ratio * h) / (ratio * 2));
				let white = new cv.Mat.zeros(_h, w, cv.CV_8U);
				// match the channel numbers of the two image
				cv.cvtColor(white, white, cv.COLOR_GRAY2BGR);
				white.data.fill(color);
				let matVector = createMatVector(white, src, white);
				// extend source image vertically
				cv.vconcat(matVector, dst);
				white.delete();
			}
		}
		
		cv.resize(dst, dst, new cv.Size(width, height), cv.INTER_LINEAR);
		if (net_chanels === 1) {
			cv.cvtColor(dst, dst, cv.COLOR_BGR2GRAY);
			dst = tf.expandDims(dst, 2);
		}
		const mat255 = cv.Mat.zeros(dst.rows, dst.cols, cv.CV_8U);
		cv.cvtColor(mat255, mat255, cv.COLOR_GRAY2BGR);
		mat255.data.fill(255);
		cv.divide(dst, mat255, dst);
		
		if (write)
			await imwrite(dst, 'models/output.png')
		src.delete();
		mat255.delete();
		return dst;
	} catch (e) {
		console.error(cvTranslateError(e));
		src.delete();
		return null;
	}
}
