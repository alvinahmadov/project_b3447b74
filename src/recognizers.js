import * as tf        from "@tensorflow/tfjs";
import {ConfigParser} from "./parser.js";
import {
	densenet,
	processImage,
	saveModelAsJSON,
	cvToTensor
}                     from "./utils.js";


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
		const input_shape = [this.parser.height,
		                     this.parser.width,
		                     this.parser.netChanels];
		
		let inputs = tf.input({shape: input_shape});
		
		let dense_layer = densenet(input_shape)
			.apply(inputs);
		
		let reshaped = tf.layers.reshape({name: 'Reshape', targetShape: [24, 128]})
			.apply(dense_layer);
		
		let gruLayer =
			tf.layers.gru({name: "GRU", units: 256, activation: 'tanh',
				              recurrentActivation: 'sigmoid', returnSequences: true, dropout: 0.2});
		
		let blstm_1 = tf.layers.bidirectional({name: 'Bidirectional', layer: gruLayer})
			.apply(reshaped);
		let blstm_2 = tf.layers.bidirectional({name: 'Bidirectional_1', layer: gruLayer})
			.apply(blstm_1);
		
		let outputs = tf.layers.dense(
			{name: "Dense_2", units: this.parser.letters.length + 1, activation: "softmax"}
		).apply(blstm_2);
		
		const model = tf.model({inputs: inputs, outputs: outputs});
		
		if (!this.debug) {
			model.summary();
			saveModelAsJSON('models/debug_model.json', model);
		}
		
		return model;
	}
	
	async predict(image_path = './models/example8.png') {
		let model = this.model;
		model.loadWeights(this.parser.modelPath, false)
		processImage(image_path, this.parser.parameters)
			.then(image => {
				const tensor = cvToTensor(image)
				const predictions = model.predict(tensor);
				
				const data = predictions.argMax(2).dataSync()
				const letters = this.parser.letters
				const letter_len = letters.length;
				let label = '' // data[0];
				
				for (let index = 0; index < data.length; ++index)
					if (index <= letter_len && index >= 0)
						label += this.parser.letters[data[index]]
				
				//TODO(Alvin): Implement CTCDecoder for nodejs
				
				if (this.debug) {
					// console.log("Predictions:");
					// console.log(predictions);
					console.log("label", label)
				}
			});
	}
}
