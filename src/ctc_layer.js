
import * as assert from "assert";


function applyOpHelper(inputs, seqLength, mergeRepeated) {}


/**
 * @param {Array[]} inputs
 *
 * */
function greedyDecoder(inputs, seqLength, mergeRepeated) {
	let numTimeSteps = inputs.length
	
	for (let i = 0; i < numTimeSteps; ++i) {
		assert(inputs[i].length === seqLength.length, "The shape of inputs does not match with the shape of the input")
	}
}





export class CTCDecoder {
	constructor(numClasses, batchSize, mergeRepeated) {
		this._numClasses = numClasses;
		this._batchSize = batchSize;
		this._mergeRepeated = mergeRepeated;
	}
	
	/**
	 * Decodes the output of a softmax.
	 *
	 * Can use either greedy search (also known as best path)
	 * or a constrained dictionary search.
	 *
	 * @param {Tensor} input `(samples, time_steps, num_categories)`
	 * containing the prediction, or output of the softmax.
	 * @param {Tensor} inputLength `(samples, )` containing the sequence length for
	 * each batch item in `y_pred`.
	 *
	 * @param {boolean} mergeRepeated
	 *
	 * @returns:
	 * if `greedy` is `true`, returns a list of one element that
	 * contains the decoded sequence.
	 * If `false`, returns the `top_paths` most probable
	 * decoded sequences.
	 *
	 * Each decoded sequence has shape (samples, time_steps).
	 * @important: blank labels are returned as `-1`.
	 * Tensor `(top_paths, )` that contains
	 * the log probability of each decoded sequence.
	 * */
	decode(input, inputLength, mergeRepeated = true) {}
}

/**
 * CTCGreedyDecoder is an implementation of the simple best path decoding
 * algorithm, selecting at each timestep the most likely class at each timestep.
 */
export class CTCGreedyDecoder extends CTCDecoder {
	constructor(num_classes, batch_size, merge_repeated)
	{
		super(num_classes, batch_size, merge_repeated);
	}
	
	
	decode(input, inputLength, mergeRepeated = true) {
		// for (let b = 0; b < this._batchSize; ++b) {
		// 	let seq_len_b = inputLength[b];
		// 	// Only writing to beam 0
		// 	let output_b = (*output)[0][b];
		//
		// 	int prev_class_ix = -1;
		// 	(*scores)(b, 0) = 0;
		// 	for (int t = 0; t < seq_len_b; ++t) {
		// 		auto row = input[t].row(b);
		// 		int max_class_ix;
		// 		(*scores)(b, 0) += -row.maxCoeff(&max_class_ix);
		// 		if (max_class_ix != Decoder::blank_index_ &&
		// 			!(Decoder::merge_repeated_ && max_class_ix == prev_class_ix)) {
		// 			output_b.push_back(max_class_ix);
		// 		}
		// 		prev_class_ix = max_class_ix;
		// 	}
		// }
	}
}

/**
 *
 * 	 * @param {boolean} greedy perform much faster best-path search if `true`.
 * This does not use a dictionary.
 * @param {number} beam_width if `greedy` is `false`: a beam search decoder will be used
 * with a beam of this width.
 * @param {number} top_paths: if `greedy` is `false`,
 * how many of the most probable paths will be returned.
 * */
