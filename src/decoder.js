/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 *
 * @see Alex Graves - Connectionist Temporal Classification: Labelling Unsegmented
                      Sequence Data with Recurrent Neural Networks [https://www.cs.toronto.edu/~graves/icml_2006.pdf]
 * */

import * as tf  from "@tensorflow/tfjs-node";
import {rowMax} from "./utils.js";

/**
 * The CTCDecoder is an abstract interface to be implemented when providing a
 * decoding method on the timestep output of a RNN trained with CTC loss.
 *
 * The two types of decoding available are:
 *   - greedy path, through the CTCGreedyDecoder
 *   - beam search, through the CTCBeamSearchDecoder
 */
class CTCDecoder {
	static topPaths = 1;
	
	/**
	 * @param {boolean} mergeRepeated boolean
	 * @param {boolean} debug
	 * */
	constructor(mergeRepeated = true, debug) {
		this.mergeRepeated_ = mergeRepeated;
		this.results = new Map()
		this.logProb = null
		this.debug = debug;
	}
	
	/**
	 *  Dimensionality of the input/output is expected to be:
	 *  - seq_len[b] - b = 0 to batch_size_
	 *  - input[t].rows(b) - t = 0 to timesteps; b = 0 t batch_size_
	 *  - output.size() specifies the number of beams to be returned.
	 *  - scores(b, i) - b = 0 to batch_size; i = 0 to output.size()
	 */
	decode(input, inputLength) {}
	
	get logProbability() {
		return this.logProb;
	}
	
	getIndices() {
		return this._getSetHelper('indices');
	}
	
	setIndices(indice) {
		this._getSetHelper('indices', indice);
	}
	
	getValues() {
		return this._getSetHelper('values');
	}
	
	setValues(value) {
		this._getSetHelper('values', value);
	}
	
	getShape() {
		return this._getSetHelper('shape');
	}
	
	setShape(shape) {
		this._getSetHelper('shape', shape);
	}
	
	_getSetHelper(key, t = null) {
		if (t === null || t === undefined)
			return this.results.get(key);
		
		if (!this.results.has(key))
			this.results.set(key, [t]);
		else
			this.results.get(key).push(t);
	}
}

// noinspection JSPrimitiveTypeWrapperUsage
/**
 * CTCGreedyDecoder is an implementation of the simple best path decoding
 * algorithm, selecting at each timestep the most likely class at each timestep.
 * */
export class CTCGreedyDecoder extends CTCDecoder {
	/**
	 * @param {boolean} mergeRepeated. Default: True.
	 * @param {boolean} debug.
	 * */
	constructor(mergeRepeated = true, debug = true) {
		super(mergeRepeated, debug);
	}
	
	//public
	/**
	 * Performs greedy decoding on the logits given in input (best path).
	 *
	 * Note: Regardless of the value of mergeRepeated, if the maximum index of a
	 * given time and batch corresponds to the blank index `(numClasses - 1)`, no
	 * new element is emitted.
	 *
	 * If `mergeRepeated` is `true`, merge repeated classes in output.
	 *
	 * @param {Tensor} inputs: 3-D `float` `Tensor` sized `[maxTime, batchSize, numClasses]`.
	 * The logits.
	 * @param {Tensor} sequenceLength: 1-D `int32` vector containing sequence lengths, having
	 * size `[batch_size]`.
	 * */
	async decode(inputs, sequenceLength) {
		const inputsShape = inputs.shape;
		const maxTime = inputsShape[0];
		const batchSize = inputsShape[1];
		const numClasses = inputsShape[2];
		let blankIndex = numClasses - 1;
		
		if (batchSize !== sequenceLength.shape[0])
			throw Error("sequenceLength.length != batch_size");
		
		let logProb = tf.buffer([batchSize, CTCGreedyDecoder.topPaths], "float32");
		let inputListTimesteps = Array();
		let inputsData = inputs.dataSync();
		
		// Group pretrained data into equal-sized slices as tensor2d [batchSize, numClasses],
		// group count is `maxTime` and each group has `numClasses` elements
		for (let timeStep = 0, step = 0; timeStep < maxTime;
		     ++timeStep, step = timeStep * batchSize * numClasses) {
			let sliced = inputsData.slice(step, (timeStep + 1) * batchSize * numClasses)
			if (sliced.length > 0)
				inputListTimesteps.push(tf.tensor2d(sliced, [batchSize, numClasses]));
		}
		
		let seqLenT = sequenceLength.arraySync();
		let sequences = [[]];
		
		let decoder = (begin, end) => {
			for (let idx = begin; idx < end; ++idx) {
				sequences[idx] = new Array();
				let sequence = [];
				let prevIndices = -1;
				for (let t = 0; t < seqLenT[idx]; ++t) {
					let maxClassIndices;
					const rmax = rowMax(inputListTimesteps[t], idx);
					let prob = logProb.get(idx, 0);
					logProb.set(prob + (-rmax[0]), idx, 0);
					maxClassIndices = rmax[1];
					if (maxClassIndices !== blankIndex &&
						!(this.mergeRepeated_ && maxClassIndices === prevIndices)) {
						sequence.push(maxClassIndices);
					}
					prevIndices = maxClassIndices;
				}
				sequences[idx].push(sequence);
			}
		}
		
		decoder(0, batchSize - 1);
		this._save(sequences);
		
		if (this.debug) {
			console.log("Sequences: ", sequences);
			console.log("Log prob : ", logProb.get(0, 0));
		}
		this.logProb = logProb.toTensor();
		inputs.dispose();
		sequenceLength.dispose();
	}
	
	//private
	_save(sequences) {
		const batchSize = sequences.length;
		const topPaths = CTCGreedyDecoder.topPaths;
		let numEntries = Array(topPaths).fill(0);
		
		for (let sequence of sequences) {
			if (sequence.length !== topPaths)
				throw Error(`sequence.length !== topPaths (${sequence.length}, ${topPaths})`);
			
			for (let path = 0; path < topPaths; ++path) {
				numEntries[path] += sequence[path].length;
				if (this.debug){
					console.log("Path:      ", path);
					console.log("NumEntrs:  ", numEntries[path]);
				}
			}
		}
		
		for (let path = 0; path < topPaths; ++path) {
			const pNum = numEntries[path];
			let indices = tf.buffer([pNum, 2], "int32");
			let values = tf.buffer([pNum], "int32");
			let shape = tf.buffer([2], "float32");
			
			let maxDecoded = 0;
			let offset = 0;
			
			for (let b = 0; b < batchSize; ++b) {
				let batch = sequences[b][path];
				let numDecoded = batch.length;
				maxDecoded = Math.max(maxDecoded, numDecoded);
				
				if (numDecoded > 0) {
					if (offset > values.size)
						throw Error("Offset should be smaller than values.size()");
					
					for (let k = 0; k < numDecoded; ++k) {
						values.set(batch[k], offset + k);
					}
				}
				for (let t = 0; t < numDecoded; ++t, ++offset) {
					indices.set(b, offset, 0);
					indices.set(t, offset, 1);
				}
			}
			shape.set(batchSize, 0);
			shape.set(maxDecoded, 1);
			
			if (this.debug) {
				console.log("BatchSize: ", batchSize);
			}
			
			this.setIndices(indices.toTensor());
			this.setValues(values.toTensor());
			this.setShape(shape.toTensor());
		}
	}
}
