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
	 * */
	constructor(mergeRepeated = true) {
		this.mergeRepeated_ = mergeRepeated;
		this.indices_ = null
		this.values_ = null
		this.shape_ = null
	}
	
	/**
	 *  Dimensionality of the input/output is expected to be:
	 *  - seq_len[b] - b = 0 to batch_size_
	 *  - input[t].rows(b) - t = 0 to timesteps; b = 0 t batch_size_
	 *  - output.size() specifies the number of beams to be returned.
	 *  - scores(b, i) - b = 0 to batch_size; i = 0 to output.size()
	 */
	decode(input, inputLength) {}
	
	get indices() {
		return this.indices_;
	}
	
	get values() {
		return this.values_;
	}
	
	get shape() {
		return this.shape_;
	}
}

/**
 * CTCGreedyDecoder is an implementation of the simple best path decoding
 * algorithm, selecting at each timestep the most likely class at each timestep.
 * */
export class CTCGreedyDecoder extends CTCDecoder {
	/**
	 * @param {boolean} mergeRepeated. Default: True.
	 * */
	constructor(mergeRepeated = true) {
		super(mergeRepeated);
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
	decode(inputs, sequenceLength) {
		const inputsShape = inputs.shape;
		const maxTime = inputsShape[0];
		const batchSize = inputsShape[1];
		const numClasses = inputsShape[2];
		let blankIndex = numClasses - 1;
		
		if (batchSize !== sequenceLength.shape[0])
			throw Error("sequenceLength.length != batch_size");
		
		let logProb = tf.buffer([batchSize, this.topPaths], "float32");
		let inputListTimesteps = Array();
		let inputsData = inputs.dataSync();
		
		// Group pretrained data into equal-sized slices as tensor2d [batchSize, numClasses],
		// group count is `maxTime` and each group has `numClasses` elements
		for (let timeStep = 0, step = 0;
		     timeStep < maxTime, step !== inputsData.length;
		     ++timeStep, step = timeStep * batchSize * numClasses) {
			let sliced = inputsData.slice(step,
			                              (timeStep + 1) * batchSize * numClasses)
			if (sliced.length > 0)
				inputListTimesteps.push(tf.tensor2d(sliced, [batchSize, numClasses]));
		}
		
		let seqLenT = sequenceLength.arraySync();
		let sequences = Array(batchSize);
		
		let decoder = (begin, end) => {
			for (let idx = begin; idx < end; ++idx) {
				sequences[idx] = Array();
				let prevIndices = -1;
				for (let t = 0; t < seqLenT[idx]; ++t) {
					let maxClassIndices;
					let rowMax = rowMax(inputListTimesteps[t], idx);
					logProb.set(-rowMax[0], idx, 0);
					maxClassIndices = rowMax[1];
					if (maxClassIndices !== blankIndex &&
						!(this.mergeRepeated_ && maxClassIndices === prevIndices)) {
						sequences[idx][0].push(maxClassIndices);
					}
					prevIndices = maxClassIndices;
				}
			}
		}
		// Run decoder() in threads of 50 * maxTime * numClasses
		this._save(sequences);
	}
	
	//private
	_save(sequences) {
		const batchSize = sequences.length;
		const topPaths = CTCDecoder.topPaths;
		
		let numEntries = Array(topPaths)
		for (let batch_s of sequences)
			for (let path = 0; path < topPaths; ++path)
				numEntries[path] += batch_s[path].length;
		
		for (let path = 0; path < topPaths; ++path) {
			const pNum = numEntries[path];
			
			let indices = tf.TensorBuffer([pNum, 2], "float32")
			let values = tf.TensorBuffer([pNum], "int32")
			let shape = tf.TensorBuffer([2], "float32")
			
			let maxDecoded = 0;
			let offset = 0;
			
			for (let b = 0; b < batchSize; ++b) {
				let batch = sequences[b][path];
				let numDecoded = batch.length;
				maxDecoded = Math.max(maxDecoded, numDecoded);
				if (numDecoded > 0) {
					if (offset > values.size())
						throw Error("Offset should be smaller than values_t.size()");
				}
				for (let t = 0; t < numDecoded; ++t, ++offset) {
					indices.set(b, offset, 0);
					indices.set(t, offset, 1);
				}
			}
			
			shape.set(batchSize, 0);
			shape.set(maxDecoded, 1);
		}
	}
}
