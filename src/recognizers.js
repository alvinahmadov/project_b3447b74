import * as tf            from "@tensorflow/tfjs-node";
import {ConfigParser}     from "./parser.js";
import {CTCGreedyDecoder} from "./decoder.js";
import {Lambda}           from "./lambda_layer.js";
import {densenet}         from "./densenet.js";
import {
	processImage,
	saveModelAsJSON,
	readWeightMaps,
	cvToTensor,
	ones,
	mulScalar,
	pathJoin
}                         from "./utils.js";
import {
	PERMUTATION,
	EPSILON,
	DATA_ROOT,
	MODEL_ROOT,
	WEIGHTS_KEY,
	DEFAULT_GRU_ARGS_T8
}                         from "./constants.js"

class Processor {
	constructor(path, type) {
		this.parser = new ConfigParser(path, type);
	}
	
	get model() {
		throw Error("Not implemented");
	}
	
	async predict(image_path, shardsPrefix) {
		throw Error("Not implemented");
	}
}

// noinspection JSUnresolvedVariable,JSUnresolvedFunction
export class ProcessorType8 extends Processor {
	/**
	 * Constructor of ProcessorType8.
	 *
	 * @param {string} path Path to the parameters file
	 * @param {boolean} debug Show debugging messages.
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
		
		let inputs = tf.input({name: "input_1", shape: inputShape});
		
		let denseNet = densenet(inputShape).apply(inputs);
		
		let squeezed = new Lambda((x) => tf.squeeze(x, 1))
			.apply(denseNet);
		
		let reshaped = tf.layers.reshape({name: 'reshape', targetShape: [24, 128]})
			.apply(squeezed);
		
		let blstm_1 = tf.layers.bidirectional({
			                                      name:      'bidirectional',
			                                      layer:     tf.layers.gru(DEFAULT_GRU_ARGS_T8),
			                                      mergeMode: "concat"
		                                      })
			.apply(reshaped);
		
		let blstm_2 = tf.layers.bidirectional({
			                                      name:      'bidirectional_1',
			                                      layer:     tf.layers.gru(DEFAULT_GRU_ARGS_T8),
			                                      mergeMode: "concat"
		                                      })
			.apply(blstm_1);
		
		let outputs = tf.layers.dense({
			                              name:       "dense",
			                              units:      this.parser.letters.length + 1,
			                              activation: "softmax"
		                              })
			.apply(blstm_2);
		
		const model = tf.model({inputs: inputs, outputs: outputs});
		
		if (this.debug) {
			model.summary();
			saveModelAsJSON(pathJoin(DATA_ROOT, MODEL_ROOT, 'debug_model.json'), model);
		}
		
		return model;
	}
	
	async predict(image_path, shardsPrefix) {
		let result = "";
		const model = this.model
		
		try {
			const image = await processImage(image_path, this.parser.parameters);
			
			return readWeightMaps(this.parser.modelJSON[WEIGHTS_KEY],
			                      shardsPrefix).then((weightsMap) => {
				const imgTensor = cvToTensor(image);
				model.loadWeights(weightsMap, false);
				const predictions = model.predict(imgTensor);
				let input = tf.log(tf.add(tf.transpose(predictions, PERMUTATION), EPSILON));
				let sequenceLength = mulScalar(ones(predictions.shape[0]), predictions.shape[1]);
				sequenceLength = tf.cast(sequenceLength, "int32");
				
				if (this.debug) {
					console.log("Predictions: ")
					predictions.print();
					console.log("Processed predictions:")
					input.print();
				}
				
				let decoder = new CTCGreedyDecoder(true,
				                                   this.debug);
				decoder.decode(input, sequenceLength);
				const indice = decoder.indices[0];
				const value = decoder.values[0];
				const shape = decoder.shape[0];
				
				for (const v of value.dataSync())
					if (v > 0)
						result += this.parser.letters[v];
					else
						console.log(v)
				
				indice.dispose();
				value.dispose();
				shape.dispose();
				
				return result;
			}).catch(r => console.error(r));
		} catch (e) {
			console.error(e);
		}
		return result;
		// const sample = [3, 18, 29, 7, 30, 19]
	}
}
