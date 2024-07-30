test_mistral_v1_cot:
	python supervised_ft/main.py supervised_ft/params_input/mistral_v1_cot.json

test_mistral_v2_cot:
	python supervised_ft/main.py supervised_ft/params_input/mistral_v2_cot.json

test_vistral_fc:
	python supervised_ft/main.py supervised_ft/params_input/vistral_fc.json

test_vistral_fc_cot:
	python supervised_ft/main.py supervised_ft/params_input/vistral_fc_cot.json