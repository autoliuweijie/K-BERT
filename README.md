# K-BERT
![](https://img.shields.io/badge/license-MIT-000000.svg)

Sorce code and datasets for ["K-BERT: Enabling Language Representation with Knowledge Graph"](https://arxiv.org/abs/1909.07606v1), which is implemented based on the [UER](https://github.com/dbiir/UER-py) framework.


## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```


## Prepare

* Download the ``google_model.bin`` from [here](https://share.weiyun.com/5GuzfVX), and save it to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://share.weiyun.com/5BvtHyO), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://share.weiyun.com/5Id9PVZ), unzip and place them in the ``datasets/`` directory.

The directory tree of K-BERT:
```
K-BERT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── datasets
│   ├── book_review
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── chnsenticorp
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│    ...
│
├── models
│   ├── google_config.json
│   ├── google_model.bin
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
├── requirements.txt
├── run_kbert_cls.py
└── run_kbert_ner.py
```


## K-BERT for text classification

### Classification example

Run example on Book review with CnDbpedia:
```sh
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/book_review/train.tsv \
    --dev_path ./datasets/book_review/dev.tsv \
    --test_path ./datasets/book_review/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
    > ./outputs/kbert_bookreview_CnDbpedia.log &
```

Results:
```
Best accuracy in dev : 88.80%
Best accuracy in test: 87.69%
```

Options of ``run_kbert_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
        [--output_model_path] - Path to the output model.
```

### Classification benchmarks

Accuracy (dev/test %) on different dataset:

| Dataset       | HowNet       | CnDbpedia     |
| :-----        | :----:       | :----:        |
| Book review   | 88.75/87.75  | 88.80/87.69   |
| ChnSentiCorp  | 95.00/95.50  | 94.42/95.25   |
| Shopping      | 97.01/96.92  | 96.94/96.73   |
| Weibo         | 98.22/98.33  | 98.29/98.33   |
| LCQMC         | 88.97/87.14  | 88.91/87.20   |
| XNLI          | 77.11/77.07  | 76.99/77.43   |


## K-BERT for named entity recognization (NER)

### NER example

Run an example on the msra_ner dataset with CnDbpedia:

```
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_ner.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/msra_ner/train.tsv \
    --dev_path ./datasets/msra_ner/dev.tsv \
    --test_path ./datasets/msra_ner/test.tsv \
    --epochs_num 5 --batch_size 16 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_msraner_CnDbpedia.bin \
    > ./outputs/kbert_msraner_CnDbpedia.log &
```

Results:
```
The best in dev : precision=0.957, recall=0.962, f1=0.960
The best in test: precision=0.953, recall=0.959, f1=0.956
```

Options of ``run_kbert_ner.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph.
        [--output_model_path] - Path to the output model.
```


## K-BERT for domain-specific tasks

Experimental results on domain-specific tasks (Precision/Recall/F1 %):

| KG            | Finance_QA         | Law_QA              | Finance_NER        | Medicine_NER        |
| :-----        | :----:             | :----:              | :----:             | :----:              |
| HowNet        |  0.805/0.888/0.845 | 0.842/0.903/0.871   | 0.860/0.888/0.874  | 0.935/0.939/0.937   |
| CN-DBpedia    |  0.814/0.881/0.846 | 0.814/0.942/0.874   | 0.860/0.887/0.873  | 0.935/0.937/0.936   |
| MedicalKG     | --                 | --                  | --                 | 0.944/0.943/0.944   |


## Acknowledgement

This work is a joint study with the support of Peking University and Tencent Inc.

If you use this code, please cite this paper:
```
@inproceedings{weijie2019kbert,
  title={{K-BERT}: Enabling Language Representation with Knowledge Graph},
  author={Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang},
  booktitle={Proceedings of AAAI 2020},
  year={2020}
}
```


