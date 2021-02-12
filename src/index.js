import {ProcessorType8} from "./recognizers.js";
import {DATA_ROOT}      from "./constants.js"
import {pathJoin}       from "./utils.js";

async function recognize(config_path, image_path, shardsPrefix) {
	let proc = new ProcessorType8(config_path, false);
	return await proc.predict(image_path, shardsPrefix);
}

recognize('params.yaml',
          pathJoin(DATA_ROOT, "example8.png"),
          'data/models/type8')
	.then(t => console.log("Text:", t))
	.catch(reason => console.error(reason));
