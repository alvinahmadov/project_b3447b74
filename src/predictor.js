/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 * */

import * as tf            from "@tensorflow/tfjs-node";
import {ConfigParser}     from "./parser.js";
import {CTCGreedyDecoder} from "./decoder.js";
import {Lambda}           from "./lambda_layer.js";
import {densenet}         from "./densenet.js";
import {
	processImage,
	saveModelAsJSON,
	readWeightMaps,
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
	PTYPE
}                         from "./constants.js"

class PredictorBase {
	constructor(path, type, debug) {
		this.parser = new ConfigParser(path, type);
		this.debug = debug;
	}
	
	/**
	 * Must be implemented in derived classes
	 * */
	get model() {
		throw Error("Not implemented");
	}
	
	get shape() {
		return [this.parser.height,
		        this.parser.width,
		        this.parser.netChannels];
	}
	
	createModel(inputs, output, args) {
		try {
			const units = (args.units !== undefined) ? args.units : 256;
			const dropout = (args.dropout !== undefined) ? args.dropout : 0.2;
			const activation = (args.activation !== undefined) ? args.activation : 'tanh';
			const recurrentActivation = (args.recurrentActivation !== undefined) ? args.recurrentActivation : 'sigmoid';
			const kernelInitializer = (args.kernelInitializer !== undefined) ? args.kernelInitializer : 'glorotUniform';
			const gruName1 = (args.name1 !== undefined) ? args.name1 : 'gru';
			const gruName2 = (args.name2 !== undefined) ? args.name2 : 'gru_1';
			
			const gru1 = tf.layers.gru({
				                           name:                gruName1,
				                           units:               units,
				                           dtype:               'float32',
				                           returnSequences:     true,
				                           kernelInitializer:   kernelInitializer,
				                           activation:          activation,
				                           recurrentActivation: recurrentActivation,
				                           dropout:             dropout,
				                           implementation:      2,
				                           resetAfter:          false
			                           })
			
			const gru2 = tf.layers.gru({
				                           name:                gruName2,
				                           units:               units,
				                           dtype:               'float32',
				                           returnSequences:     true,
				                           kernelInitializer:   kernelInitializer,
				                           activation:          activation,
				                           recurrentActivation: recurrentActivation,
				                           dropout:             dropout,
				                           implementation:      2,
				                           resetAfter:          false
			                           })
			
			let blstm_1 = tf.layers.bidirectional(
				{
					name:      'bidirectional_1',
					layer:     gru1,
					dtype:     'float32',
					mergeMode: 'concat'
				}).apply(output);
			
			let blstm_2 = tf.layers.bidirectional(
				{
					name:      'bidirectional_2',
					layer:     gru2,
					dtype:     'float32',
					mergeMode: 'concat'
				}).apply(blstm_1);
			
			let denseLayer = tf.layers.dense({
				                                 name:       'dense',
				                                 dtype:      'float32',
				                                 units:      this.parser.classes.length + 1,
				                                 activation: 'softmax'
			                                 }).apply(blstm_2);
			
			const model = tf.model({inputs: inputs, outputs: denseLayer});
			
			if (this.debug) {
				model.summary();
				saveModelAsJSON(pathJoin(DATA_ROOT, MODEL_ROOT, `debug_model_${this.parser.type}.json`), model);
			}
			
			return model;
		} catch (e) {
			console.error(e)
			throw Error(e)
		}
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
		const model = this.model;
		const image = await processImage(imagePath, this.parser.parameters);
		if (image === null)
			throw Error(`Image ${imagePath} is null`);
		try {
			return readWeightMaps(this.parser.modelJSON[WEIGHTS_KEY], shardsPrefix).then((weightsMap) => {
				model.loadWeights(weightsMap);
				const predictions = model.predict(image);
				
				if (this.debug) {
					console.log("Predictions: ");
					predictions.print();
				}
				
				let input = predictions;
				if (input.shape.length < PERMUTATION.length) {
					input = predictions.expandDims(0);
				}
				
				const sequenceLength = tf.cast(
					mulScalar(ones(input.shape[0]), input.shape[1]),
					"int32"
				);
				
				input = tf.log(tf.add(tf.transpose(input, PERMUTATION), tf.scalar(EPSILON)));
				predictions.dispose();
				
				const decoder = new CTCGreedyDecoder(true, this.debug);
				
				return decoder.decode(input, sequenceLength).then(() => {
					const indice = decoder.getIndices()[0];
					const value = decoder.getValues()[0];
					const shape = decoder.getShape()[0];
					
					for (const v of value.dataSync())
						if (v >= 0)
							result += this.parser.classes[v];
					
					if (this.debug) {
						console.log(`Indices     : [${indice.dataSync().join(', ')}]`);
						console.log(`Values      : [${value.dataSync().join(', ')}]`);
						console.log(`Shape       : [${shape.dataSync().join(', ')}]`);
						console.log(`Probability : ${decoder.logProbability.dataSync()}`);
					}
					
					indice.dispose();
					value.dispose();
					shape.dispose();
					model.dispose();
					return result;
				});
			}).catch(r => console.error(r));
		} catch (e) {
			console.error(e);
		}
		return result;
	}
}

class PredictorType1 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T1, debug);
	}
	
	get model() {
		let inputs = tf.layers.input({shape: this.shape});
		
		let conv_1 = tf.layers.conv2d({
			                              filters:    32,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same'
		                              }).apply(inputs);
		
		let pool_1 = tf.layers.maxPool2d({
			                                 poolSize: [4, 2],
			                                 strides:  2
		                                 }).apply(conv_1);
		
		let conv_2 = tf.layers.conv2d({
			                              filters:    32,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same',
		                              }).apply(pool_1);
		
		let pool_2 = tf.layers.maxPool2d({
			                                 poolSize: [4, 2],
			                                 strides:  2,
			                                 padding:  'valid'
		                                 }).apply(conv_2);
		
		let conv_3 = tf.layers.conv2d({
			                              filters:    64,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same'
		                              }).apply(pool_2);
		
		let conv_4 = tf.layers.conv2d({
			                              filters:    64,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same'
		                              }).apply(conv_3);
		
		let pool_4 = tf.layers.maxPool2d({
			                                 poolSize: [2, 1],
			                                 padding:  'same'
		                                 }).apply(conv_4);
		
		let conv_5 = tf.layers.conv2d({
			                              filters:    128,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same'
		                              }).apply(pool_4);
		
		let batchNorm = tf.layers.batchNormalization().apply(conv_5);
		let conv_6 = tf.layers.conv2d({
			                              filters:    128,
			                              kernelSize: [3, 3],
			                              activation: 'relu',
			                              padding:    'same'
		                              }).apply(batchNorm);
		
		let batchNorm_6 = tf.layers.batchNormalization().apply(conv_6);
		let pool_6 = tf.layers.maxPool2d({
			                                 poolSize: [4, 1],
			                                 padding:  'same'
		                                 }).apply(batchNorm_6);
		
		let conv_7 = tf.layers.conv2d({
			                              filters:    128,
			                              kernelSize: [2, 2],
			                              activation: 'relu'
		                              }).apply(pool_6);
		
		let squeezed = new Lambda(x => tf.squeeze(x, 1)).apply(conv_7);
		
		return this.createModel(inputs, squeezed,
		                        {
			                        name1:               'gru_1a',
			                        name2:               'gru_1b',
			                        units:               128,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid',
			                        dropout:             0.2
		                        });
	}
}

class PredictorType3 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T3, debug);
	}
	
	get model() {
		let inputs = tf.layers.input({shape: this.shape});
		
		let denseNetLayer = densenet(this.shape).apply(inputs);
		let reshaped = tf.layers.reshape({targetShape: [48, 128]}).apply(denseNetLayer);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_3a',
			                        name2:               'gru_3b',
			                        units:               256,
			                        dropout:             0.5,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
}

class PredictorType4 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T4, debug);
	}
	
	get model() {
		let inputs = tf.layers.input({shape: this.shape});
		
		let denseNetLayer = densenet(this.shape).apply(inputs);
		let squeezed = new Lambda(x => tf.squeeze(x, 1)).apply(denseNetLayer);
		let reshaped = tf.layers.reshape({targetShape: [12, 512]}).apply(squeezed);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_4a',
			                        name2:               'gru_4b',
			                        units:               256,
			                        returnSequences:     true,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid',
			                        dropout:             0.5,
		                        });
	}
}

class PredictorType5 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T5, debug);
	}
	
	get model() {
		let inputs = tf.input({shape: this.shape});
		
		let denseNet = densenet(this.shape).apply(inputs);
		
		let conv2 = tf.layers.conv2d({
			                             name:       'conv2d_Conv2D1',
			                             filters:    1024,
			                             kernelSize: [2, 2],
			                             activation: 'relu'
		                             }).apply(denseNet);
		
		let squeeze = new Lambda(
			(x) => tf.squeeze(x, 1), {name: 'lambda', dtype: 'float32', trainable: true}
		).apply(conv2);
		
		let reshaped = tf.layers.reshape({name: 'reshape', targetShape: [16, 256], dtype: 'float32'})
			.apply(squeeze);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_5a',
			                        name2:               'gru_5b',
			                        units:               256,
			                        dropout:             0.5,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
}

class PredictorType6 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T6, debug);
	}
	
	get model() {
		let inputs = tf.input({shape: this.shape});
		
		let denseNet = densenet(this.shape).apply(inputs);
		
		const squeezed = new Lambda(
			(x) => tf.squeeze(x, 1),
			{name: 'lambda', dtype: 'float32', trainable: true}
		).apply(denseNet);
		
		const reshaped = tf.layers.reshape({name: 'reshape', targetShape: [24, 128], dtype: 'float32'})
			.apply(squeezed);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_6a',
			                        name2:               'gru_6b',
			                        units:               256,
			                        dropout:             0.2,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
}

class PredictorType7 extends PredictorBase {
	constructor(path, debug = false) {
		super(path, PTYPE.T7, debug);
	}
	
	get model() {
		let inputs = tf.input({shape: this.shape});
		
		let denseNet = densenet(this.shape).apply(inputs);
		
		const squeezed = new Lambda(
			(x) => tf.squeeze(x, 1),
			{name: 'lambda', dtype: 'float32', trainable: true}
		).apply(denseNet);
		
		const reshaped = tf.layers.reshape({name: 'reshape', targetShape: [24, 128], dtype: 'float32'})
			.apply(squeezed);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_7a',
			                        name2:               'gru_7b',
			                        units:               256,
			                        dropout:             0.2,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
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
	get model() {
		let inputs = tf.input({shape: this.shape});
		
		let denseNet = densenet(this.shape).apply(inputs);
		
		const squeezed = new Lambda(
			(x) => tf.squeeze(x, 1),
			{name: 'lambda', dtype: 'float32', trainable: true}
		).apply(denseNet);
		
		const reshaped = tf.layers.reshape({name: 'reshape', targetShape: [24, 128], dtype: 'float32'})
			.apply(squeezed);
		
		return this.createModel(inputs, reshaped,
		                        {
			                        name1:               'gru_8a',
			                        name2:               'gru_8b',
			                        units:               256,
			                        dropout:             0.2,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
}

class PredictorRecap extends PredictorBase {
	/**
	 * Constructor of PredictorRecap.
	 *
	 * @param {string} path Path to the parameters file
	 * @param {boolean} debug Show debugging messages.
	 */
	constructor(path, debug = false) {
		super(path, PTYPE.RECAP, debug);
	}
	
	/**
	 * Get the pretrained model.
	 */
	get model() {
		let inputs = tf.input({shape: this.shape})
		let denseNet = densenet(null, inputs);
		denseNet.trainable = false;
		
		let output = tf.layers.globalAveragePooling2d({name: 'avg_pool'}).apply(denseNet.output);
		output = tf.layers.batchNormalization({name: 'batch_norm_1'}).apply(output);
		output = tf.layers.dense({
			                         name:       'dense_1',
			                         units:      128,
			                         activation: 'relu'
		                         }).apply(output);
		
		output = tf.layers.dense({
			                         name:       'dense_2',
			                         units:      128,
			                         activation: 'relu'
		                         }).apply(output);
		
		output = tf.layers.dropout({rate: 0.2, name: 'top_dropout'}).apply(output);
		
		output = tf.layers.dense({
			                         name: 'dense',
			                         units: this.parser.classes.length,
			                         activation: 'softmax'}).apply(output);
		
		const model = tf.model({inputs: inputs, outputs: output});
		
		if (this.debug) {
			model.summary();
			saveModelAsJSON(pathJoin(DATA_ROOT, MODEL_ROOT, `debug_model_${this.parser.type}.json`), model);
		}
		
		return model;
	}
}

class Predictor {
	constructor(path, debug = false) {
		this.path = path;
		this.debug = debug;
		this.predictor = null
	}
	
	_prepare(type) {
		switch (type) {
			case PTYPE.T1:
				this.predictor = new PredictorType1(this.path, this.debug);
				break;
			case PTYPE.T3:
				this.predictor = new PredictorType3(this.path, this.debug);
				break;
			case PTYPE.T4:
				this.predictor = new PredictorType4(this.path, this.debug);
				break;
			case PTYPE.T5:
				this.predictor = new PredictorType5(this.path, this.debug);
				break;
			case PTYPE.T6:
				this.predictor = new PredictorType6(this.path, this.debug);
				break;
			case PTYPE.T7:
				this.predictor = new PredictorType7(this.path, this.debug);
				break;
			case PTYPE.T8:
				this.predictor = new PredictorType8(this.path, this.debug);
				break;
			case PTYPE.RECAP:
				this.predictor = new PredictorRecap(this.path, this.debug);
				break;
			default:
				throw Error("Undefined type")
		}
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
		this._prepare(type);
		let res = await this.predictor.predict(imagePath, shardsPrefix);
		delete this.predictor;
		return res;
	}
}

export {Predictor}
