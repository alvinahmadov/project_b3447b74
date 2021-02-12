import {Predictor} from "./predictor.js";
import {pathJoin}  from "./utils.js";
import {
	DATA_ROOT,
	MODEL_ROOT,
	PTYPE
}                  from "./constants.js";

async function recognize(type, configPath, imagePath, shardsPrefix) {
	let pred = new Predictor(configPath, false);
	return await pred.run(type, imagePath, shardsPrefix);
}

const TYPE = PTYPE.T8;
const CONFIG = 'params.yaml';
const IMG_PATH = pathJoin(DATA_ROOT, "example8.png");
const SHARDS_PREF = pathJoin(DATA_ROOT, MODEL_ROOT, TYPE);

recognize(TYPE, CONFIG, IMG_PATH, SHARDS_PREF)
	.then(t => console.log("Text:", t))
	.catch(reason => console.error(reason));
