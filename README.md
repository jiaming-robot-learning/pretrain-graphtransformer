## Pre-training Graph Transformer

### Clone the repository
```
git clone https://github.com/jiaming-robot-learning/pretrain-graphtransformer.git
```

### Install required dependencies
```
pip install -r requirements.txt
```

### Download dataset
Download dataset from
[https://www.kaggle.com/jiaminggogogo/pretrain-graphtransformer].

Unzip and place the downloaded dataset at pretrain-graphtransformer/dataset


### Pretraining
```bash
chmod +x pretrain.sh
./pretrain.sh
```

### Finetuning
```bash
chmod +x finetune.sh
./finetune.sh
```

