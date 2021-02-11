import {ProcessorType8} from "./recognizers.js";

async function recognize() {
	let proc = new ProcessorType8('params.yaml', true);
	let text = '';
	
	try {
		text = await proc.predict('./models/example8.png');
	} catch (e) {
		console.error(e);
	}
	return text;
}

recognize().then(
	text => console.log("Text:", text)
)
