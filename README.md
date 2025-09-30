<div align="center">

  <h2><b> Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps </b></h2>
  <h4> A Fine-Grained Visual Reasoning Benchmark: ReasonMap</h4>

</div>


<div align="center">

![](https://img.shields.io/github/stars/fscdc/ReasonMap?color=yellow)
![](https://img.shields.io/github/forks/fscdc/ReasonMap?color=lightblue)
![](https://img.shields.io/github/last-commit/fscdc/ReasonMap?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
<a href="https://arxiv.org/abs/2505.18675" target="_blank"><img src="https://img.shields.io/badge/arXiv-2505.18675-009688.svg" alt="arXiv"></a>

</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2505.18675">arXiv</a>]** **[<a href="https://huggingface.co/datasets/FSCCS/ReasonMap">Dataset</a>]** **[<a href="https://x.com/si_feng32704/status/1927186378900533309">X</a>]**

</div>


---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our [research](#citation)!

<p align="center">
<img src="assets/overview.svg" width = "95%" alt="" align=center />
</p>


## Updates

- 2025-05-15: 🚀 We released evaluation code and our [website](https://fscdc.github.io/ReasonMap/) online!
- 2025-05-15: 🚀 We released [ReasonMap](https://huggingface.co/datasets/FSCCS/ReasonMap)!


## Usage

### 1. Install dependencies

If you face any issues with the installation, please feel free to open an issue. We will try our best to help you.

```bash
conda env create -f reasonmap-py310.yaml
```

### 2. Download the dataset
You can download the dataset from [HuggingFace](https://huggingface.co/datasets/FSCCS/ReasonMap).


### 3. Evaluation
You can evaluate the model performance on ReasonMap by running the following command:
```bash
# open-source models
bash script/run.sh
# closed-source models
bash script/run-closed-models.sh

# after running the above scripts, you can analyze the results by:
python cal_metrics.py
```

## Citation
If you find this benchmark useful in your research, please consider citing our paper:

```bibtex
@article{feng2025can,
  title={Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps},
  author={Feng, Sicheng and Wang, Song and Ouyang, Shuyi and Kong, Lingdong and Song, Zikai and Zhu, Jianke and Wang, Huan and Wang, Xinchao},
  journal={arXiv preprint arXiv:2505.18675},
  year={2025}
}
```