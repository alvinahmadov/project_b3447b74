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
	PTYPE,
	DEFAULT_GRU_ARGS_T8
}                         from "./constants.js"

class PredictorBase {
	constructor(path, type, debug) {
		this.parser = new ConfigParser(path, type);
		this.debug = debug;
	}
	
	/**
	 * Must be implemented in derived classes
	 * */
	model() {
		throw Error("Not implemented");
	}
	
	/**
	 * Main logic is same for all types of input data
	 *
	 * @param {string} imagePath
	 * @param {string} shardsPrefix
	 *
	 * @returns {Promise<string>}
	 * */
	async predict(imagePath, shardsPrefix) {
		let result = "";
		const model = this.model()
		
		try {
			const image = await processImage(imagePath, this.parser.parameters);
			
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

class PredictorType1 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T1, debug);
	}
	
	model() {}
}

class PredictorType3 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T3, debug);
	}
	
	model() {}
}

class PredictorType4 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T4, debug);
	}
	
	model() {}
}

class PredictorType5 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T5, debug);
	}
	
	model() {}
}

class PredictorType8 extends PredictorBase {
	/**
	 * Constructor of PredictorType8.
	 *
	 * @param {string} path Path to the parameters file
	 * @param {boolean} debug Show debugging messages.
	 */
	constructor(path, debug = false) {
		super(path, PTYPE.T8, debug);
	}
	
	/**
	 * Get the pretrained model.
	 */
	model() {
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
}

class Predictor {
	constructor(path, debug) {
		this.path = path;
		this.debug = debug;
	}
	
	/**
	 * Run predictions
	 *
	 * @param {string} type Type of prediction.
	 * @param {string} imagePath Path to the image to be processed.
	 * @param {string} shardsPrefix Prefix to the location of shard files (*.bin)
	 * @returns {Promise<string>} Recognized text
	 * */
	async run(type, imagePath, shardsPrefix) {
		let predictor;
		switch (type) {
			case PTYPE.T1:
				predictor = new PredictorType1(this.path, this.debug);
				break;
			case PTYPE.T3:
				predictor = new PredictorType3(this.path, this.debug);
				break;
			case PTYPE.T4:
				predictor = new PredictorType4(this.path, this.debug);
				break;
			case PTYPE.T5:
				predictor = new PredictorType5(this.path, this.debug);
				break;
			case PTYPE.T6:
			case PTYPE.T7:
			case PTYPE.T8:
				predictor = new PredictorType8(this.path, this.debug);
				break;
			default:
				throw Error("Undefined type")
		}
		
		return predictor.predict(imagePath, shardsPrefix);
	}
}

export {Predictor}
