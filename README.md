# MSINet
Official implementation of "[MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID](https://arxiv.org/abs/2303.07065)". 

## Datasets

Put the datasets into `./data`

* VeRi-776

```bash
python train.py
```

```bash
python train.py --pretrained
```

```bash
python train.py -ds market1501 -dt msmt17 --pretrained --epochs 250
```

### Conduct Search for Other Re-ID Datasets

```bash
python search.py -ds market1501
```

