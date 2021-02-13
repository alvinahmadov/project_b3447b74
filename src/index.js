/**
 * Alvin Ahmadov [https://github.com/AlvinAhmadov]
 * */

import {Predictor} from "./predictor.js";
import {pathJoin}  from "./utils.js";
import {
	DATA_ROOT,
	MODEL_ROOT,
	PTYPE
}                  from "./constants.js";

const TYPE = PTYPE.T8;
const CONFIG = 'params.yaml';
const IMG_PATH = pathJoin(DATA_ROOT, "example8.png");
const SHARDS_PREF = pathJoin(DATA_ROOT, MODEL_ROOT, TYPE);

(async () => {
	let pred = new Predictor(CONFIG, false);
	let text = await pred.run(TYPE, IMG_PATH, SHARDS_PREF);
	console.log("Decoded text:", text);
	return text;
})()
