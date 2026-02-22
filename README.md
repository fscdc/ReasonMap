<div align="center">

  <h2><b> [CVPR 2026] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps </b></h2>
  <h4> The first benchmark using real-world metro maps </h4>

</div>


<div align="center">

![](https://img.shields.io/github/stars/fscdc/ReasonMap?color=yellow)
![](https://img.shields.io/github/forks/fscdc/ReasonMap?color=lightblue)
![](https://img.shields.io/github/last-commit/fscdc/ReasonMap?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
<a href="https://arxiv.org/abs/2505.18675" target="_blank"><img src="https://img.shields.io/badge/arXiv-2505.18675-009688.svg" alt="arXiv"></a>
[![Dataset](https://img.shields.io/badge/🤗%20Huggingface-Dataset-yellow)](https://huggingface.co/collections/FSCCS/reasonmap)


</div>

<div align="center">

**[<a href="https://huggingface.co/papers/2505.18675">HuggingFace Daily Paper</a>]** **[<a href="https://x.com/si_feng32704/status/1927186378900533309">Twitter</a>]** **[<a href="https://mp.weixin.qq.com/s/sPJLQtHgl5DZghWLWa_H3Q">量子位</a>]**

</div>

This repository is for our paper:
> **[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](https://arxiv.org/abs/2505.18675)** \
> [Sicheng Feng](https://fscdc.github.io/)<sup>1,2,^</sup>, [Song Wang](https://songw-zju.github.io/)<sup>3,2,^</sup>, [Shuyi Ouyang](https://scholar.google.com/citations?user=pQgChLEAAAAJ&hl=zh-CN)<sup>3,2</sup>, [Lingdong Kong](https://ldkong.com/)<sup>2</sup>, [Zikai Song](https://skyesong38.github.io/)<sup>4,2</sup>, [Jianke Zhu](https://person.zju.edu.cn/en/jkzhu)<sup>3</sup>, [Huan Wang](https://huanwang.tech/)<sup>1,*</sup>, [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)<sup>2</sup> \
> <sup>1</sup>Westlake University, Hangzhou, China \
> <sup>2</sup>National University of Singapore, Singapore \
> <sup>3</sup>Zhejiang University, Hangzhou, China \
> <sup>4</sup>Huazhong University of Science and Technology, Wuhan, China \
> <sup>^</sup>Equal contribution, <sup>∗</sup>Corresponding author: wanghuan@westlake.edu.cn


<!-- ---

<div align="center">

💡 Interested in evaluating your model on ReasonMap or ReasonMap-Plus?
 
📩 Contact us [fscnkucs@gmail.com](fscnkucs@gmail.com)

</div> -->

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our [research](#citation)!

<p align="center">
<img src="assets/overview.png" width = "95%" alt="" align=center />
</p>


## Updates

- 2026-02-21: 🚀 Our paper was accepted by CVPR 2026! Thanks to all contributors!
- 2026-01-26: 🚀 The following research ([RewardMap](https://github.com/fscdc/RewardMap)) has been accepted by ICLR 2026!
- 2025-09-30: 🚀 We released [ReasonMap-Plus](https://huggingface.co/datasets/FSCCS/ReasonMap-Plus) for the following research - [RewardMap](https://github.com/fscdc/RewardMap)!
- 2025-05-15: 🚀 We released evaluation code and our [website](https://fscdc.github.io/ReasonMap/) online!
- 2025-05-15: 🚀 We released [ReasonMap](https://huggingface.co/datasets/FSCCS/ReasonMap)!


## Usage

### 1. Install dependencies

If you face any issues with the installation, please feel free to open an issue. We will try our best to help you.

```bash
conda env create -f reasonmap-py310.yaml
```

### 2. Download the dataset
You can download [ReasonMap](https://huggingface.co/datasets/FSCCS/ReasonMap) and [ReasonMap-Plus](https://huggingface.co/datasets/FSCCS/ReasonMap-Plus) from HuggingFace.


### 3. Evaluation
You can evaluate the model performance on ReasonMap by running the following command:
```bash
## ReasonMap Evaluation
# open-source models
bash script/run.sh
# closed-source models
bash script/run-closed-models.sh

## ReasonMap-Plus Evaluation
bash script/run_plus.sh

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

# further research
@article{feng2025rewardmap,
  title={RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning},
  author={Feng, Sicheng and Tuo, Kaiwen and Wang, Song and Kong, Lingdong and Zhu, Jianke and Wang, Huan},
  journal={arXiv preprint arXiv:2510.02240},
  year={2025}
}
```
