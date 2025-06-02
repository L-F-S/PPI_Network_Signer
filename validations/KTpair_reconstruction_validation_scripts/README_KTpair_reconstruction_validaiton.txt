
# Using TLM data (see TLM validation)

1. Generate a dictionary of shortest paths with signal scores

	PARALLELgenerate_SP_scores.py for KO data (WARNING: requires having features for edges of the whole PPI network)
	generate_SP_scores_TLM_inputs for TLM data based on the anat network

2. calculate score:
	reconstruct_KT_pairs.py
