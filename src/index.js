import {ProcessorType8} from "./recognizers.js";

async function recognize(config_path, image_path) {
	let proc = new ProcessorType8(config_path, false);
	let text = '';
	
	try {
		text = await proc.predict(image_path);
	} catch (e) {
		console.error(e);
	}
	return text;
}

recognize('./models/params.yaml',
          './models/example8.png')
	.then(text => console.log("Text:", text));
