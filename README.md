# Beyond Distribution Shift: Spurious Features Through the Lens of Training Dynamics

### [Main Paper](https://openreview.net/pdf?id=Tkvmt9nDmB) | [arXiv](https://arxiv.org/abs/2302.09344#:~:text=Deep%20Neural%20Networks%20(DNNs)%20are,applied%20to%20safety%2Dcritical%20domains.) | [Video](https://www.youtube.com/watch?v=kkQ0IKukx5o&ab_channel=NihalMurali) | [Poster](https://pitt-my.sharepoint.com/:b:/g/personal/nim123_pitt_edu/EYh3rVX8nOlKseTkgmx8qiYBQrC7DNpnxfnFJ5d9aJ9m2w?e=d3sVmC) | [Slides](https://pitt-my.sharepoint.com/:p:/r/personal/nim123_pitt_edu/Documents/TMLR23_Dynamics_of_Spurious_Features/TMLR_official_slides(only).pptx?d=w68b380b344094588a317687275dadf39&csf=1&web=1&e=b2M6Wj) | [Talk](https://www.youtube.com/watch?v=6pP8YQX5cmc&ab_channel=ComputationalGenomicsSummerInstituteCGSI)

Official PyTorch implementation of the [TMLR](https://jmlr.org/tmlr/) paper: <br/>
**Beyond Distribution Shift: Spurious Features Through the Lens of Training Dynamics** <br/>
[Nihal Murali<sup>1</sup>](https://scholar.google.co.in/citations?user=LVcXV4oAAAAJ&hl=en),
[Aahlad Puli<sup>3</sup>](https://gatechke.github.io/),
[Ke Yu<sup>1</sup>](https://gatechke.github.io/),
[Rajesh Ranganath<sup>3</sup>](https://gatechke.github.io/),
[Kayhan Batmanghelich<sup>2</sup>](https://www.batman-lab.com/)
<br/>
<sup>1</sup> University of Pittsburgh (ISP), <sup>2</sup> Boston University (ECE), <sup>3</sup> New York University (CS) <br/>

## Table of Contents

1. [Objective](#objective)
2. [Environment setup](#environment-setup)
3. [Downloading data](#downloading-data)
    * [(a) Downloading vision and skin data](#a-downloading-vision-and-skin-data)
    * [(b) Downloading MIMIC-CXR](#b-downloading-mimic-cxr)
4. [Training pipeline](#training-pipleline)
    * [(a) Running MoIE](#a-running-moie)
    * [(b) Compute the performance metrics](#b-compute-the-performance-metrics)
    * [(c) Validating the concept importance](#c-validating-the-concept-importance)
5. [How to Cite](#how-to-cite)
6. [License and copyright](#license-and-copyright)

## Objective

Deep Neural Networks (DNNs) are prone to learning spurious features that correlate with the label during training but are irrelevant to the learning problem. This hurts model
generalization and poses problems when deploying them in safety-critical applications. This paper aims to better understand the effects of spurious features through the lens of the
learning dynamics of the internal neurons during the training process. We make the following observations: (1) While previous works highlight the harmful effects of spurious features on
the generalization ability of DNNs, we emphasize that not all spurious features are harmful. Spurious features can be “benign” or “harmful” depending on whether they are “harder”
or “easier” to learn than the core features for a given model. This definition is model and dataset dependent. (2) We build upon this premise and use instance difficulty methods (like
Prediction Depth (Baldock et al., 2021)) to quantify “easiness” for a given model and to identify this behavior during the training phase. (3) We empirically show that the harmful
spurious features can be detected by observing the learning dynamics of the DNN’s early layers. In other words, easy features learned by the initial layers of a DNN early during the
training can (potentially) hurt model generalization. We verify our claims on medical and vision datasets, both simulated and real, and justify the empirical success of our hypothesis
by showing the theoretical connections between Prediction Depth and information-theoretic concepts like V-usable information (Ethayarajh et al., 2021). Lastly, our experiments show
that monitoring only accuracy during training (as is common in machine learning pipelines) is insufficient to detect spurious features. We, therefore, highlight the need for monitoring
early training dynamics using suitable instance difficulty metrics.


## Environment setup

```bash
conda env create --name TMLR23_spurious_dynamics -f environment.yml
conda activate TMLR23_spurious_dynamics
```

## Downloading data


| Dataset      | URL                                                                                        |
|--------------|--------------------------------------------------------------------------------------------|
| NIH          | [Kaggle Link](https://www.kaggle.com/datasets/nih-chest-xrays/data)                        |        
| MIMIC-CXR    | [PhysioNet portal](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)                     |
| CheXpert     | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)              |
| GitHub-COVID | [covid-chestxray-dataset GitHub](https://github.com/ieee8023/covid-chestxray-dataset)      |
| NICO Dataset | [Dropbox Link] (https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0) |

## Data preprocessing


| Link  | Description                        |
|-----------|------------------------------------|
| [nih_full.csv](https://drive.google.com/file/d/1qdYMBL1zhpm2CyOohi8Id5qCHZxvVLH1/view?usp=drive_link)  | full nih data    | 

## Training pipeline




## How to Cite
* TMLR23 Main Paper
```
@article{murali2023shortcut,
  title={Shortcut Learning Through the Lens of Early Training Dynamics},
  author={Murali, Nihal and Puli, Aahlad Manas and Yu, Ke and Ranganath, Rajesh and Batmanghelich, Kayhan},
  journal={arXiv preprint arXiv:2302.09344},
  year={2023}
}
```

* Shortcut paper published in Workshop on Spurious Correlations, Invariance and Stability, ICML 2023
```
@article{muralishortcut,
  title={Shortcut Learning Through the Lens of Training Dynamics},
  author={Murali, Nihal and Puli, Aahlad Manas and Yu, Ke and Ranganath, Rajesh and Batmanghelich, Kayhan}
}
```

## License and copyright

Licensed under the [MIT License](LICENSE)

Copyright © [Batman Lab](https://www.batman-lab.com/), 2023

