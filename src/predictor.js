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
	PTYPE, DATA_FORMAT
}                         from "./constants.js"

class PredictorBase {
	constructor(path, type, debug) {
		this.parser = new ConfigParser(path, type);
		this.debug = debug;
		this.modelName = 'model';
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
		const units = (args.units !== undefined) ? args.units : 256;
		const dropout = (args.dropout !== undefined) ? args.dropout : 0.2;
		const activation = (args.activation !== undefined) ? args.activation : 'tanh';
		const recurrentActivation = (args.recurrentActivation !== undefined) ? args.recurrentActivation : 'sigmoid';
		const kernelInitializer = (args.kernelInitializer !== undefined) ? args.kernelInitializer : 'glorotUniform';
		
		let blstm_1 = tf.layers.bidirectional(
			{
				layer:     tf.layers.gru(
					{
						name:                'gru',
						units:               units,
						dtype:               'float32',
						returnSequences:     true,
						kernelInitializer:   kernelInitializer,
						activation:          activation,
						recurrentActivation: recurrentActivation,
						dropout:             dropout,
						implementation:      2,
						resetAfter:          false
					}
				),
				dtype:     'float32',
				mergeMode: 'concat'
			}).apply(output);
		
		let blstm_2 = tf.layers.bidirectional(
			{
				layer:     tf.layers.gru(
					{
						name:                'gru_1',
						units:               units,
						dtype:               'float32',
						returnSequences:     true,
						kernelInitializer:   kernelInitializer,
						activation:          activation,
						recurrentActivation: recurrentActivation,
						dropout:             dropout,
						implementation:      2,
						resetAfter:          false
					}),
				dtype:     'float32',
				mergeMode: 'concat'
			}).apply(blstm_1);
		
		let denseLayer = tf.layers.dense({
			                                 dtype:      'float32',
			                                 units:      this.parser.letters.length + 1,
			                                 activation: 'softmax'
		                                 }).apply(blstm_2);
		
		const model = tf.model({name: this.modelName, inputs: inputs, outputs: denseLayer});
		
		if (this.debug) {
			model.summary();
			saveModelAsJSON(pathJoin(DATA_ROOT, MODEL_ROOT, `debug_model_${this.parser.type}.json`), model);
		}
		return model;
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
				
				let input = tf.log(tf.add(tf.transpose(predictions, PERMUTATION), tf.scalar(EPSILON)));
				let sequenceLength = mulScalar(ones(predictions.shape[0]), predictions.shape[1]);
				sequenceLength = tf.cast(sequenceLength, "int32");
				predictions.dispose()
				
				let decoder = new CTCGreedyDecoder(true, this.debug);
				
				return decoder.decode(input, sequenceLength).then(() => {
					const indice = decoder.getIndices()[0];
					const value = decoder.getValues()[0];
					const shape = decoder.getShape()[0];
					
					for (const v of value.dataSync())
						if (v >= 0)
							result += this.parser.letters[v];
					
					if (this.debug) {
						console.log(`Indices     : [${indice.dataSync().join(', ')}]`);
						console.log(`Values      : [${value.dataSync().join(', ')}]`);
						console.log(`Shape       : [${shape.dataSync().join(', ')}]`);
						console.log(`Probability : ${(100 * decoder.logProbability.dataSync()[0]).toFixed()}%`);
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
		this.modelName = 'model1';
	}
	
	get model() {
		console.log(this.shape)
		let inputs = tf.layers.input({shape: this.shape});
		
		let conv_1 = tf.layers.conv2d({
			                              filters:    32,
			                              kernelSize: [3, 3],
			                              strides:    [1, 1],
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
			                              strides:    [1, 1],
			                              activation: 'relu',
			                              padding:    'same',
		                              }).apply(pool_1);
		
		let pool_2 = tf.layers.maxPool2d({
			                                 poolSize: [4, 2],
			                                 strides:  [2, 2],
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
		
		return this.createModel(inputs, squeezed, 'model1',
		                        {
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
		this.modelName = 'model3';
	}
	
	get model() {
		let inputs = tf.layers.input({shape: this.shape});
		
		let denseNetLayer = densenet(this.shape).apply(inputs);
		let reshaped = tf.layers.reshape({targetShape: [48, 128]}).apply(denseNetLayer);
		
		return this.createModel(inputs, reshaped, 'model3',
		                        {
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
		this.modelName = 'model4';
	}
	
	get model() {
		let inputs = tf.layers.input({shape: this.shape});
		
		let denseNetLayer = densenet(this.shape).apply(inputs);
		let squeezed = new Lambda(x => tf.squeeze(x, 1)).apply(denseNetLayer);
		let reshaped = tf.layers.reshape({targetShape: [12, 512]}).apply(squeezed);
		
		return this.createModel(inputs, reshaped, 'model4',
		                        {
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
		this.modelName = 'model5';
	}
	
	get model() {
		let inputs = tf.input({name: "input5", shape: this.shape});
		
		let denseNet = densenet(this.shape).apply(inputs);
		
		let conv2 = tf.layers.conv2d({
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
		this.modelName = 'model6';
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
		this.modelName = 'model7';
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
		this.modelName = 'model8';
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
			                        units:               256,
			                        dropout:             0.2,
			                        activation:          'tanh',
			                        recurrentActivation: 'sigmoid'
		                        });
	}
}

class Predictor {
	constructor(path, debug) {
		this.path = path;
		this.debug = debug;
	}
	
	_prepare(type) {
		switch (type) {
			case PTYPE.T1:
				return new PredictorType1(this.path, this.debug);
			case PTYPE.T3:
				return new PredictorType3(this.path, this.debug);
			case PTYPE.T4:
				return new PredictorType4(this.path, this.debug);
			case PTYPE.T5:
				return new PredictorType5(this.path, this.debug);
			case PTYPE.T6:
				return new PredictorType6(this.path, this.debug);
			case PTYPE.T7:
				return new PredictorType7(this.path, this.debug);
			case PTYPE.T8:
				return new PredictorType8(this.path, this.debug);
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
		try {
			let predictor = this._prepare(type);
			return predictor.predict(imagePath, shardsPrefix);
		} catch (e) {
			console.error(e);
		}
	}
}

export {Predictor}
