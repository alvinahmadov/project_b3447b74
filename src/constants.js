export const AXIS = 3;
export const PERMUTATION = [1, 0, 2];
export const BLOCKS = [6, 12, 24, 16];
export const EPSILON = 1e-7;
export const DATA_ROOT = 'data'
export const MODEL_ROOT = 'models'
export const WEIGHTS_KEY = 'weightsManifest'

export const DEFAULT_GRU_ARGS_T8 = {
	name:                'GRU',
	units:               256,
	activation:          'tanh',
	recurrentActivation: 'hardSigmoid',
	kernelInitializer:   'glorotUniform',
	biasInitializer:     'zeros',
	returnSequences:     true,
	dropout:             0.2
}

