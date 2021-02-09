import * as tf from "@tensorflow/tfjs";

const EPSILON = 1e-7;

// noinspection JSUnusedGlobalSymbols
/**
 * The CTCDecoder is an abstract interface to be implemented when providing a
 * decoding method on the timestep output of a RNN trained with CTC loss.
 *
 * The two types of decoding available are:
 *   - greedy path, through the CTCGreedyDecoder
 *   - beam search, through the CTCBeamSearchDecoder
 */
class CTCDecoder {
	/**
	 * @param {number} numClasses
	 * @param {number} batchSize
	 * @param {boolean} mergeRepeated
	 * */
	constructor(numClasses, batchSize, mergeRepeated) {
		this.numClasses_ = numClasses;
		this.batchSize_ = batchSize;
		this.mergeRepeated_ = mergeRepeated;
		this.blankIndex_ = numClasses - 1;
	}
	
	/**
	 *  Dimensionality of the input/output is expected to be:
	 *  - seq_len[b] - b = 0 to batch_size_
	 *  - input[t].rows(b) - t = 0 to timesteps; b = 0 t batch_size_
	 *  - output.size() specifies the number of beams to be returned.
	 *  - scores(b, i) - b = 0 to batch_size; i = 0 to output.size()
	 */
	decode(input, inputLength){}
}

/**
 * @extends CTCDecoder
 * CTCGreedyDecoder is an implementation of the simple best path decoding
 * algorithm, selecting at each timestep the most likely class at each timestep.
 * */
export class CTCGreedyDecoder extends CTCDecoder {
	/**
	 * @param {number} numClasses
	 * @param {number} batchSize
	 * @param {boolean} mergeRepeated
	 * */
	constructor(numClasses, batchSize, mergeRepeated) {
		super(numClasses, batchSize, mergeRepeated);
	}
	
	decode2(input, inputLength) {
	
	}
	
	decode(input, inputLength, output, scores){
		if (output === undefined || output[0].length < this.batchSize_)
			throw Error("output needs to be of size at least (1, batch_size).");
		
		if (scores.rows < this.batchSize_ || scores.cols === 0)
			return Error("scores needs to be of size at least (batch_size, 1).");
		
		input = tf.log(tf.add(tf.transpose(input, [1, 0, 2]), EPSILON));
		inputLength = tf.cast(inputLength, 'int32');
		
		// For each batch entry, identify the transitions
		for (let b = 0; b < this.batchSize_; ++b) {
			let seqLenB = inputLength[b];
			// Only writing to beam 0
			let outputB = output[0][b];
			
			let prev_class_ix = -1;
			scores[b][0] = 0; // (b, 0)
			for (let t = 0; t < seqLenB; ++t) {
				let row = input[t].row(b);
				let max_class_ix = 0;
				scores[b][0] += -row.maxCoeff(max_class_ix);
				if (max_class_ix !== this.blankIndex_ &&
					!(this.mergeRepeated_ && max_class_ix === prev_class_ix)) {
					outputB.push_back(max_class_ix);
				}
				prev_class_ix = max_class_ix;
			}
		}
		return true;
	}
}
