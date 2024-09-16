# Exploring Flan-T5 for Post-ASR Error Correction

This repository contains the code and experiments for the paper "Exploring Flan-T5 for Post-ASR Error Correction". The project investigates the use of the Flan-T5 model for Generative Speech Error Correction (GenSEC) to enhance ASR outputs by mapping n-best hypotheses into a single, accurate transcription.

> [!IMPORTANT]  
> All the code for running the experiments is available in this repository. The pre-trained models are available on the Hugging Face model hub. [Open an issue](https://github.com/MorenoLaQuatra/FlanEC/issues/new) if you find any issue or need help.

## How to run the code

This repository contains the code for training and evaluating the FLANEC models. Before starting, you may want to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Training and Evaluation

The `config/` folder contains the configuration files for training and evaluation of all model types (e.g., standard-ft and LoRA) and sizes (e.g., `base`, `large`, `xl`).

To train a model, you may want to look at the specific configuration file:
- `config/flan_t5_base.yaml` for the base model with standard fine-tuning 🔥
- `config/flan_t5_large.yaml` for the large model with standard fine-tuning 🔥
- `config/flan_t5_xl.yaml` for the xl model with standard fine-tuning 🔥
- `config/flan_t5_base_lora.yaml` for the base model with LoRA 🧊
- `config/flan_t5_large_lora.yaml` for the large model with LoRA 🧊
- `config/flan_t5_xl_lora.yaml` for the xl model with LoRA 🧊

To train a model, you can run the following command:

- **LoRA** - base model example:
```bash
python train_flanec_lora.py --config config/flan_t5_base_lora.yaml
```

- **Standard fine-tuning** - base model example:
```bash
python train_flanec.py --config config/flan_t5_base.yaml
```

To evaluate a model, you may want to look at the specific configuration file and then run the following command:

- **LoRA** - base model example:
```bash
python infer_flanec_lora.py --config config/flan_t5_base_lora.yaml
```

- **Standard fine-tuning** - base model example:
```bash
python infer_flanec.py --config config/flan_t5_base.yaml
```

The evaluation script prints on screen the WER and CER scores computed with the `evaluate` library from Hugging Face. It also print the WER (Edit Distance) computed according to the official [Hypo2Trans evaluation script](https://github.com/Hypotheses-Paradise/Hypo2Trans/blob/ce9d088e92323e0d558cdc84dbf636c642d45835/H2T-LoRA/inference.py#L29). The results reported in the paper are computed with the official script (even if WER is slightly higher than the one computed with the `evaluate` library).

## Pre-trained models

The models trained with Cumulative Dataset (CD) settings are available on the Hugging Face model hub:
- [flanec-base-cd](https://huggingface.co/morenolq/flanec-base-cd)
- [flanec-large-cd](https://huggingface.co/morenolq/flanec-large-cd)
- [flanec-xl-cd](https://huggingface.co/morenolq/flanec-xl-cd)
- [flanec-base-cd-lora](https://huggingface.co/morenolq/flanec-base-cd-lora)
- [flanec-large-cd-lora](https://huggingface.co/morenolq/flanec-large-cd-lora)
- [flanec-xl-cd-lora](https://huggingface.co/morenolq/flanec-xl-cd-lora)

The models trained with Single Dataset (SD) settings are not yet available. This is because there is a model for each dataset in [HyPoradise](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html) dataset, thus there should be 8 (datasets) x 3 (model sizes) x 2 (model types) = 48 models. If you are interested in these models, please [open an issue](https://github.com/MorenoLaQuatra/FlanEC/issues/new) to let us know.

If you want to use pre-trained models to reproduce the results, you can use the following code snippet:

```bash
python infer_flanec_lora.py --config config/flan_t5_base_lora.yaml --inference.specific_test_file <path-to-test-json>
```

### Inference prompt format

To generate the correct trascription from a list of hypothesis, you want to use the same prompt used during training. The prompt should be a JSON file with the following format:

```txt
Generate the correct transcription for the following n-best list of ASR hypotheses:
1. <hypothesis-1>
2. <hypothesis-2>
...
5. <hypothesis-5>
```

The model should be able to generate the correct transcription from the list of hypotheses. For more information please refer to [our paper - missing for the moment, will be updated upon publication](#).

## Acknowledgements

- GenSEC Challenge: https://sites.google.com/view/gensec-challenge/home
- Hypo2Trans: https://github.com/Hypotheses-Paradise/Hypo2Trans

## Citation

If you use this code or the models in your research, please cite the following paper:

**the official citation will be updated upon SLT 2024 proceedings publication**

```bibtex
@inproceedings{laquatra2024flanec,
  title={FLANEC: Exploring FLAN-T5 for Post-ASR Error Correction},
  author={La Quatra, Moreno and Salerno, Valerio and Tsao, Yu and Sabato Marco, Siniscalchi},
  booktitle={Proceedings of the 2024 IEEE Workshop on Spoken Language Technology},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.