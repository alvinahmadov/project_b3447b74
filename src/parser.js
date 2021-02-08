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
			'net_channels': this.net_chanels,
			'fill_color':   this.fill_color,
			'letters':      this.letters
		};
	}
	
	get width() {
		return this.params[this.typename]['width'];
	}
	
	get height() {
		return this.params[this.typename]['height'];
	}
	
	get netChanels() {
		return this.params[this.typename]['net_chanels'];
	}
	
	get fillColor() {
		return this.params[this.typename]['fill_color'];
	}
	
	get letters() {
		return this.params[this.typename]['letters'];
	}
	
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
