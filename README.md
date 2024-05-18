# VagueGAN: A GAN-Based Data Poisoning Attack Against Federated Learning Systems

Code for the 

SECON 2023 paper: [VagueGAN: A GAN-Based Data Poisoning Attack Against Federated Learning Systems](https://ieeexplore.ieee.org/document/10287523)

## Reference

[Data Poisoning Attacks Against Federated Learning Systems](https://github.com/git-disl/DataPoisoning_FL)

## Dependencies

1) Create a virtualenv (Python 3.7)
2) Install dependencies inside of virtualenv (```pip install -r requirements.pip```)
3) If you are planning on using pca defense, you will need to install ```matplotlib```. This is not required for running experiments, and is not included in the requirements file
4) We retain the interface for label flipping attacks. If you plan to reproduce the label flipping attack, you will need to modify ```VagueGAN_attack.py```

## Instructions for execution

We outline the steps required to execute different experiments below.

### Configuration

1) ```python generate_data_distribution.py``` This downloads the datasets, as well as generates a static distribution of the training and test data to provide consistency in experiments.
2) ```python generate_default_models.py``` This generates an instance of all of the models used in the paper, and saves them to disk.

### Basic USAGE

- Federated learning system hyperparameters can be set in the ```federated_learning/arguments.py``` file
- VagueGAN hyperparameters can be set in the ```federated_learning/utils/main.py``` file
- Most specific experiment settings are located in the respective experiment files (see the following sections)

### VagueGAN Data Piosoning Attack

Running VagueGAN attack: ```python VagueGAN_attack.py```

### Unsupervised Variant of VagueGAN

Set in the ```federated_learning/utils/main.py```

### VagueGAN Attack Federated Learning System with Non-iid Data

Set in the ```sever.py```

### Three-dimensional local model visualization

Set in the ```pca_defense.py```

### PCA Defense Against Label Flipping Attacks

Running PCA defense: ```python pca_defense.py```

### MCD Defense Against Data Poisoning Attacks

Running MCD defense: ```python MCD_detection_metrics.py```

## Poisoned Data 

### VagueGAN

<img src="[https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/VagueGAN_poisoned_data.png" alt="Alt text" width="80%" /><br><br>

![image](https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/VagueGAN_poisoned_data.png)

### Unsupervised Variant of VagueGAN

![image](https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/usVagueGAN_poisoned_data.png)

## Poisoned Model

![image](https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/poisoned_model1.png)

![image](https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/poisoned_model2.png)

![image](https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure/tree/master/example/poisoned_model3.png)

## Citing

If you find this code helpful to your research, please cite our attack or defense paper:

```
@INPROCEEDINGS{10287523,
  author={Sun, Wei and Gao, Bo and Xiong, Ke and Lu, Yang and Wang, Yuwei},
  booktitle={2023 20th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)}, 
  title={VagueGAN: A GAN-Based Data Poisoning Attack Against Federated Learning Systems}, 
  year={2023},
  volume={},
  number={},
  pages={321-329},
  keywords={Computer aided instruction;Federated learning;Distance learning;Training data;Generative adversarial networks;Data models;Sensors;Fedrated learning(FL);Security and Privacy;Generative Adversarial Networks(GAN)},
  doi={10.1109/SECON58729.2023.10287523}}

```

or

```
@INPROCEEDINGS{10287523,
  author={Sun, Wei and Gao, Bo and Xiong, Ke and Lu, Yang and Wang, Yuwei},
  booktitle={2023 20th Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)}, 
  title={VagueGAN: A GAN-Based Data Poisoning Attack Against Federated Learning Systems}, 
  year={2023},
  volume={},
  number={},
  pages={321-329},
  keywords={Computer aided instruction;Federated learning;Distance learning;Training data;Generative adversarial networks;Data models;Sensors;Fedrated learning(FL);Security and Privacy;Generative Adversarial Networks(GAN)},
  doi={10.1109/SECON58729.2023.10287523}}

```
