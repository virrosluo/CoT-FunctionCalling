# Finetuning Vistral 7B Chat

Due to the private nature of the dataset used for finetuning Vistral 7B Chat, we cannot share it here.

## Hyperparameter Tuning

You can modify any hyperparameters in the training JSON configuration file located in the `supervised_ft/params_input` folder.

## Running the Training Process

To finetune the model, you can run each experiment using the Makefile with the following options:

1. **Finetuning Mistral 7B v0.1 on Chain of Thought dataset**
    ```bash
    test_mistral_v1_cot
    ```

2. **Finetuning Mistral 7B v0.2 Chat on Chain of Thought dataset**
    ```bash
    test_mistral_v2_cot
    ```

3. **Finetuning Vistral 7B Chat on Vietnamese Function Calling dataset**
    ```bash
    test_vistral_fc
    ```

4. **Enhancing Function Calling with Chain of Thought method on Vistral 7B Chat for Vietnamese dataset**
    ```bash
    test_vistral_fc_cot
    ```

---