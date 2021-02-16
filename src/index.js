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

const CONFIG = 'params.yaml';
const PARAMS = [
	{
		type:         PTYPE.T1,
		imagePath:    pathJoin(DATA_ROOT, 'example1.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T1)
	},
	{
		type:         PTYPE.T3,
		imagePath:    pathJoin(DATA_ROOT, 'example3.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T3)
	},
	{
		type:         PTYPE.T4,
		imagePath:    pathJoin(DATA_ROOT, 'example4.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T4)
	},
	{
		type:         PTYPE.T5,
		imagePath:    pathJoin(DATA_ROOT, 'example5.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T5)
	},
	{
		type:         PTYPE.T6,
		imagePath:    pathJoin(DATA_ROOT, 'example6.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T6)
	},
	{
		type:         PTYPE.T7,
		imagePath:    pathJoin(DATA_ROOT, 'example7.jpg'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T7)
	},
	{
		type:         PTYPE.T8,
		imagePath:    pathJoin(DATA_ROOT, 'example8.png'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.T8)
	},
	{
		type:         PTYPE.RECAP,
		imagePath:    pathJoin(DATA_ROOT, 'example9.jpg'),
		shardsPrefix: pathJoin(DATA_ROOT, MODEL_ROOT, PTYPE.RECAP)
	}
];

async function run(type, imagePath, shardsPrefix) {
	let pred = new Predictor(CONFIG, false);
	let text = await pred.run(type, imagePath, shardsPrefix);
	console.log("Decoded:", text);
	return text;
}

(async () => {
	for (const param of PARAMS) {
		console.log(`RUNNING ${param.type.toUpperCase()}`);
		await run(param.type, param.imagePath, param.shardsPrefix);
	}
})();
