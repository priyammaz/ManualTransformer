# Manual Transformer

Autograd makes you lazy! Lets build a Transformer completely manually! This is a precursor to [MyTorch](https://github.com/priyammaz/MyTorch/tree/main) which aims to be a fully Autograd based system. Although autograd is powerful, we can (and should) be performing known composites of operations manually for efficiency! This is also an opportunity to understand all the formulas (forward and backward) for all the different ops we commonly come across in Transformers!

### Inspiration

If you love [micrograd](https://github.com/karpathy/micrograd) and [minGPT](https://github.com/karpathy/minGPT) by [Karpathy](https://github.com/karpathy) this is the next step! We want to build  GPT model with absolutely no existing Deep Learning Frameworks, so we understand exactly how it all actually happens behind the scenes!

### Installation

```bash
pip install cupy numpy tqdm
```

### Train a Tiny GPT

```bash
python train.py
```

This will store some weights and some tokenizer files in the ```work_dir``` directory. 

### Generate Some Text

This will just keep generating a bunch of text so you can see some poorly generates Shakespeare!

```bash
python inference.py
```


### Derivations

You will find all of the main derivations for each operation in [```derivations.pdf```]() if you want to see the details of why we do all this!


### To-Do

- [ ] Create a self contained Jupyter Notebook in colab with all the derivations nicely in Latex!