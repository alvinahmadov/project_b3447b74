import {ProcessorType8} from "./recognizers.js";

async function recognize() {
	let proc = new ProcessorType8('params.yaml')
	
	try {
		return proc.predict('./models/example8.png');
	} catch (e) {
		console.error(e)
	}
}

recognize().then(
	text => console.log("Text:", text)
)
