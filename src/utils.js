/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 * */
import * as tf     from "@tensorflow/tfjs-node";
import * as fs     from "fs";
import jsYaml      from "js-yaml";
import canvas      from "canvas"
import jsonfile    from "jsonfile";
import {JSDOM}     from "jsdom";
import {cv}        from "opencv-wasm";

const {loadImage, createCanvas, Canvas, Image, ImageData} = canvas;

export function assert(expr, msg) {
	if (!expr) {
		throw new Error(typeof msg === 'string' ? msg : msg());
	}
}

// String

export function pathJoin(...chunks) {
	var separator = '/';
	var replace = new RegExp(`${separator}{3}`, 'g');
	return chunks.join(separator).replace(replace, separator);
}

/// String
// IO

function printMap(map, lineLen) {
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

function printArray(array, lineLen) {
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

export function printData(data, lineLen = 10, name = null) {
	let out
	
	if (name !== null)
		out = name + ': '
	
	if (data instanceof Map) {
		out = printMap(data, lineLen);
	} else if (data instanceof Array) {
		out = printArray(data, lineLen)
	}
	
	console.log(out)
}

/// IO

export function toList(x) {
	if (Array.isArray(x)) {
		return x;
	}
	return [x];
}

export function readParams(file_path, defaultKey = 'models') {
	const contents = fs.readFileSync(file_path);
	return (defaultKey !== '')
	       ? jsYaml.load(contents)[defaultKey]
	       : jsYaml.load(contents);
}

// Tensorflow

export async function readWeightMaps(manifest, prefix = '') {
	const load = tf.io.weightsLoaderFactory(
		filePaths => filePaths.map(filePath => fs.readFileSync(filePath).buffer)
	);
	return load(manifest, prefix);
}

export function rowMax(tensor, rowIndex) {
	const buffer = tensor.bufferSync();
	let maxCol = 0;
	
	if (!tensor.shape[1])
		throw Error("Dimension must be 1");
	
	let maxProb = buffer.get(rowIndex, 0);
	for (let i = 0; i < tensor.shape[1]; ++i) {
		if (buffer.get(rowIndex, i) > maxProb) {
			maxProb = buffer.get(rowIndex, i);
			maxCol = i;
		}
	}
	return [maxProb, maxCol];
}

/**
 * Wrapper for tensorflow ones
 * @param {Array, number} shape
 * @returns {Tensor}
 * */
export function ones(shape) {
	if (!(shape instanceof Array))
		shape = [shape];
	return tf.ones(shape);
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

export function saveModelAsJSON(path, model) {
	jsonfile.writeFile(
		path, model.toJSON(), () => console.log("Model save finished.")
	);
}

export function loadModelFromJSON(path) {
	let object = null;
	try {
		object = jsonfile.readFileSync(path, {encoding: 'utf-8', flag: 'r'});
	} catch (e) {
		console.error(e);
	}
	return object;
}

/// Tensorflow
// Image

export async function cvRead(img_path) {
	installDOM();
	let image = await loadImage(img_path);
	return cv.imread(image);
}

export async function cvWrite(img, file_path) {
	try {
		const canvas = createCanvas(300, 300);
		cv.imshow(canvas, img);
		fs.writeFileSync(file_path, canvas.toBuffer('image/png'));
	} catch (e) {
		console.error(cvTranslateError(e));
	}
}

function cvCreateMatVector(...args) {
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

function _processHImg(src, h, w, ratio, color) {
	let dst = new cv.Mat();
	const white = new cv.Mat.ones(h, w, cv.CV_8UC3);
	white.data.fill(color);
	// match the channel numbers of the two image
	const matVector = cvCreateMatVector(src, white);
	// extend source image horizontally
	cv.hconcat(matVector, dst);
	white.delete();
	return dst;
}

function _processVImg(src, h, w, ratio, color) {
	let dst = new cv.Mat();
	const white = new cv.Mat.ones(h, w, cv.CV_8UC3);
	white.data.fill(color);
	const matVector = cvCreateMatVector(white, src, white);
	// extend source image vertically
	cv.vconcat(matVector, dst);
	white.delete();
	return dst;
}

/**
 * @param {string} imgPath Path to the image to be processed
 * @param {Object} params
 * @param {boolean} write Write the result image to filesystem
 * */
export async function processImage(imgPath,
                                   params,
                                   write = true) {
	const width = params['width'];
	const height = params['height'];
	const netChannels = params['net_channels'];
	const color = params['fill_color'];
	const ratio = Math.floor(width / height);
	try {
		let src = await cvRead(imgPath);
		let dst;
		// Turn source image to RGB reducing the number of channels
		cv.cvtColor(src, src, cv.COLOR_RGBA2BGR, 0);
		let w = src.cols;
		let h = src.rows;
		
		// Find image orientation and choose right op
		if (w / h !== ratio) {
			if (w / h < ratio) {
				dst = _processHImg(src, h, (ratio * h) - w, ratio, color);
			} else if (w / h > ratio) {
				dst = _processVImg(src, Math.floor((w - ratio * h) / (ratio * 2)), w, ratio, color);
			}
		} else dst = src;
		cv.resize(dst, dst, new cv.Size(width, height), cv.INTER_LINEAR);
		if (netChannels === 1)
			cv.cvtColor(dst, dst, cv.COLOR_BGR2GRAY);
		
		let imgTensor = tf.tensor3d(dst.data, [dst.rows, dst.cols, netChannels], "float32");
		imgTensor = tf.div(imgTensor, tf.scalar(255));
		imgTensor = tf.expandDims(imgTensor, 0);
		
		return imgTensor;
	} catch (e) {
		console.error(cvTranslateError(e));
		src.delete();
		return null;
	}
}

/// Image
