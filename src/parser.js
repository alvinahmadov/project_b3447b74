import {readParams} from "./utils.js";

export class ConfigParser {
	constructor(path = 'params.yaml', typename) {
		this.params = readParams(path);
		this.typename = typename
	}
	
	/**
	 * Get training parameters
	 * @returns {Object}
	 * */
	get parameters() {
		return {
			'width':        this.width,
			'height':       this.height,
			'net_channels': this.netChanels,
			'fill_color':   this.fillColor,
			'letters':      this.letters
		};
	}
	
	/**
	 * @returns {number}
	 * */
	get width() {
		return this.params[this.typename]['width'];
	}
	
	/**
	 * @returns {number}
	 * */
	get height() {
		return this.params[this.typename]['height'];
	}
	
	/**
	 * @returns {number}
	 * */
	get netChanels() {
		return this.params[this.typename]['net_chanels'];
	}
	
	/**
	 * @returns {number}
	 * */
	get fillColor() {
		return this.params[this.typename]['fill_color'];
	}
	
	/**
	 * @returns {string}
	 * */
	get letters() {
		return this.params[this.typename]['letters'];
	}
	
	/**
	 * @returns {number}
	 * */
	get ctcInputLength() {
		return this.params[this.typename]['ctc_input_length'];
	}
	
	get maxLabelLength() {
		return this.params[this.typename]['max_label_length'];
	}
	
	get modelPath() {
		return this.params[this.typename]['model_path'];
	}
}
