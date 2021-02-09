import * as tf        from "@tensorflow/tfjs-node";
import {ConfigParser} from "./parser.js";
// import {CTCGreedyDecoder}   from "./ctc_layer.js";
import {Lambda}       from "./lambda_layer.js";
import {
	processImage,
	saveModelAsJSON,
	cvToTensor,
	checkData,
	ones,
	mulScalar,
	registerOccurences,
	printData,
	decode
}                     from "./utils.js";

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
	constructor(path, debug = true) {
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
			console.clear();
			model.summary();
			saveModelAsJSON('models/debug_model.json', model);
		}
		
		return model;
	}
	
	async predict(image_path = './models/example8.png') {
		let model = this.model;
		model.loadWeights(this.parser.modelPath,
		                  false);
		processImage(image_path, this.parser.parameters)
			.then(image => {
				const letters = this.parser.letters;
				const lettersLen = letters.length;
				const tensor = cvToTensor(image);
				const predictions = model.predict(tensor);
				const sample = [3, 18, 29, 7, 30, 19]
				
				let input = tf.log(tf.add(tf.transpose(predictions, [1, 0, 2]), Math.E));
				let inputLength = mulScalar(ones(predictions.shape[0]), predictions.shape[1])
				inputLength = tf.cast(inputLength, 'int32');
				
				const data = input.argMax().dataSync()
				
				if (this.debug) {
					let dataOut = registerOccurences(data);
					console.log("=================================================================")
					console.log(dataOut);
					console.log("_________________________________________________________________")
					console.log(data[0]);
					console.log("=================================================================")
					console.log("Check", checkData(dataOut, sample))
					console.log("=================================================================")
					let text = ''
					
					data.forEach((x)=> {
						text += letters.charAt(x)
					})
					console.log(lettersLen, text)
				}
			});
	}
}
