import * as tf            from "@tensorflow/tfjs-node";
import {ConfigParser}     from "./parser.js";
import {CTCGreedyDecoder} from "./decoder.js";
import {
	processImage,
	saveModelAsJSON,
	cvToTensor,
	ones,
	mulScalar,
	
	PERMUTATION,
	EPSILON
}                         from "./utils.js";

import {densenet} from "./densenet.js";

class Processor {
	constructor(path, type) {
		this.parser = new ConfigParser(path, type);
	}
}

// noinspection JSUnresolvedVariable,JSUnresolvedFunction
export class ProcessorType8 extends Processor {
	/**
	 * Constructor of ProcessorType8.
	 *
	 * @param {string} path An identifier for this instance of TextData.
	 * @param {boolean} debug
	 */
	constructor(path, debug = false) {
		super(path, 'type8');
		this.debug = debug
	}
	
	/**
	 * Get the pretrained model.
	 */
	get model() {
		const inputShape = [this.parser.height,
		                    this.parser.width,
		                    this.parser.netChanels];
		
		let inputs = tf.input({shape: inputShape});
		
		let dense_layer = densenet(inputs)
			.apply(inputs);
		
		let reshaped = tf.layers.reshape({name: 'Reshape', targetShape: [24, 128]})
			.apply(dense_layer);
		
		let gruLayer =
			tf.layers.gru({
				              name:                "GRU", units: 256, activation: 'tanh',
				              recurrentActivation: 'sigmoid', returnSequences: true, dropout: 0.2
			              });
		
		let blstm_1 = tf.layers.bidirectional({name: 'Bidirectional', layer: gruLayer})
			.apply(reshaped);
		let blstm_2 = tf.layers.bidirectional({name: 'Bidirectional_1', layer: gruLayer})
			.apply(blstm_1);
		
		let outputs = tf.layers.dense(
			{name: "Dense_2", units: this.parser.letters.length + 1, activation: "softmax"}
		).apply(blstm_2);
		
		const model = tf.model({inputs: inputs, outputs: outputs});
		
		if (this.debug) {
			model.summary();
			saveModelAsJSON('models/debug_model.json', model);
		}
		
		return model;
	}
	
	async predict(image_path) {
		let model = this.model;
		let result = '';
		model.loadWeights(this.parser.modelPath, false);
		processImage(image_path, this.parser.parameters)
			.then(image => {
				const imgTensor = cvToTensor(image);
				const predictions = model.predict(imgTensor);
				let input = tf.log(tf.add(tf.transpose(predictions, PERMUTATION), EPSILON));
				let sequenceLength = mulScalar(ones(predictions.shape[0]), predictions.shape[1])
				
				try {
					let decoder = new CTCGreedyDecoder();
					decoder.decode(input, sequenceLength);
				} catch (e) {
					console.error(e);
				}
			});
		
		// const sample = [3, 18, 29, 7, 30, 19]
		return result;
	}
}
