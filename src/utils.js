import jsYaml    from "js-yaml";
import {JSDOM}   from "jsdom";
import * as http from "https"

import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import {cv}    from "opencv-wasm";
import canvas  from "canvas"

const {loadImage, createCanvas, Canvas, Image, ImageData} = canvas;

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

function _printMap(map, lineLen) {
	let res = '{ '
	let idx = 0
	const size = map.length;
	
	for (let pair of map) {
		if (idx + 1 !== size)
			res += pair[0] + ' => ' + pair[1]
		res += (((++idx) % lineLen === 0) ? ',\n' : ', ');
	}
	
	res += ' }';
	
	return res;
}

function _printArray(array, lineLen) {
	let size = array.length;
	let res = '[';
	
	for (let i = 0; i < size; ++i) {
		res += array[i].toString()
		if (i + 1 !== size)
			res += (i % lineLen + 1 === 0) ? ',\n' : ', ';
	}
	res += ']'
	
	return res
}

function toList(x) {
	if (Array.isArray(x)) {
		return x;
	}
	return [x];
}

export function getFile(url, filename) {
	const file = fs.createWriteStream('./models/' + filename);
	const request = http.get(url, (response) => {
		response.pipe(file);
		file.on('finish', function () {
			file.close();
		});
	});
	
	console.log("+++++", request)
	
	return file;
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

export function printData(data, lineLen = 10, name = null) {
	let out
	
	if (name !== null)
		out = name + ': '
	
	if (data instanceof Map) {
		out = _printMap(data, lineLen);
	} else if (data instanceof Array) {
		out = _printArray(data, lineLen)
	}
	
	console.log(out)
}

/**
 * @param {Map} dataMap
 * @param {number[]} sample
 * */
export function checkData(dataMap, sample) {
	let res = new Map()
	sample.forEach((x) => res.set(x, false))
	
	for (let pair of dataMap) {
		for (let d of sample) {
			if (d === parseInt(pair[0]))
				res.set(d, true);
		}
	}
	
	return res
}

export function decode(input, sequenceLength) {
	let set = new Set()
	if (sequenceLength.length !== 1)
		throw Error("Must be 1 element")
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

/**
 * Wrapper for tensorflow ones
 * @param {Array, number} shape
 * @returns {Tensor}
 * */
export function ones(shape) {
	if (!(shape instanceof Array))
		shape = [shape]
	return tf.ones(shape)
}

/**
 * Wrapper for tensorflow's scalar to tensor multiplication
 * without need to convert typings
 * @param {Tensor} tensor
 * @param {number, string, Uint8Array} scalar
 * @param {string} dtype
 *
 * @returns {Tensor}
 * */
export function mulScalar(tensor, scalar, dtype = "int32") {
	if (scalar instanceof Number
		|| scalar instanceof String)
		scalar = tf.scalar(scalar);
	
	return tf.mul(tensor, scalar);
}

export function registerOccurences(data) {
	let kmap = new Map()
	
	data.forEach((x) => {
		if (kmap.has(x))
			kmap.set(x, kmap.get(x) + 1);
		else
			kmap.set(x, 1);
	})
	
	return kmap
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
