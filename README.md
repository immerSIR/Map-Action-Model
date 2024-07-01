<p align="center">
  <img src="https://dashboard.map-action.com/static/media/logo.ff03b7a9.png" width="100" />
</p>
<p align="center">
    <h1 align="center">MAP-ACTION-MODEL</h1>
</p>
<p align="center">
    <em>Map your actions with precision and value!</em>
</p>

<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat-square&logo=YAML&logoColor=white" alt="YAML">
	<img src="https://img.shields.io/badge/Jinja-B41717.svg?style=flat-square&logo=Jinja&logoColor=white" alt="Jinja">
	<img src="https://img.shields.io/badge/PowerShell-5391FE.svg?style=flat-square&logo=PowerShell&logoColor=white" alt="PowerShell">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<br>
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat-square&logo=Docker&logoColor=white" alt="Docker">
	<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=flat-square&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/DVC-13ADC7.svg?style=flat-square&logo=DVC&logoColor=white" alt="DVC">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat-square&logo=FastAPI&logoColor=white" alt="FastAPI">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Running Map-Action-Model](#-running-Map-Action-Model)
>   - [ Tests](#-tests)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

Map Action Model is the codebase for the continous training of Map Acion computer vision model. [Developper Doc](https://223mapaction.github.io/Map-Action-Model/)

![Selection_080](https://github.com/223MapAction/Map-Action-Model/assets/64170643/cf467484-6f1b-49b7-9c09-e9bf9b0031ae)

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project follow a modular architecture with dependencies on libraries like Brotli, Pillow, and FastAPI. It utilizes a mix of Python and related technologies for development. |
| 🔩 | **Code Quality**  | The code quality is maintained with the use of tools such as MkDocs for documentation and potentially other linting tools based on the repository contents. |
| 📄 | **Documentation** | The project includes MKDocs for documentation generation, providing extensive and structured documentation for the codebase. |
| 🔌 | **Integrations**  | Key integrations include Brotli, tqdm, and FastAPI among others, suggesting a reliance on external libraries and tools for functionality. |
| 🧩 | **Modularity**    | The project is structured in a modular way, with dependencies on various libraries like Torch, torchvision, and others, indicating potential code reusability. |
| 🧪 | **Testing**       | Testing frameworks is PyTest |
| 📦 | **Dependencies**  | Key external libraries and dependencies include Brotli, Pillow, FastAPI, Torch, torchvision, and others, indicating a reliance on diverse libraries for functionality. |


---

##  Repository Structure

```sh
└── Map-Action-Model/
    ├── .github
    │   └── workflows
    │       ├── deploy-docs.yml
    │       ├── training-on-gpu.yml
    │       ├── unittesting.yml
    │       └── zenml_action.yml
    ├── Dockerfile._cuda
    ├── Dockerfile.fastapi
    ├── LICENCE
    ├── _cd.yml
    ├── _ci.yml
    ├── code
    │   ├── .zen
    │   │   └── config.yaml
    │   ├── TFLearning.ipynb
    │   ├── pipelines
    │   │   └── zenml_pipeline.py
    │   ├── steps
    │   │   ├── dagshub_utils
    │   │   ├── data_preprocess
    │   │   ├── model
    │   │   ├── model_eval
    │   │   ├── plot_metrics
    │   │   └── training_step
    │   ├── utilities.ipynb
    │   └── zenml_running.py
    ├── data.dvc
    ├── data_upload.py
    ├── ma_env
    │   ├── bin
    │   │   ├── Activate.ps1
    │   │   ├── activate
    │   │   ├── activate.csh
    │   │   ├── activate.fish
    │   │   ├── pip
    │   │   ├── pip3
    │   │   ├── pip3.10
    │   │   ├── python
    │   │   ├── python3
    │   │   └── python3.10
    │   ├── lib
    │   │   └── python3.10
    │   └── lib64
    │       └── python3.10
    ├── model
    │   └── main.py
    ├── requirements.txt
    ├── services
    │   └── unittesting
    │       └── Dockerfile
    └── zenml_config
        └── zenml_conf.yml
```

---

##  Modules

<details closed><summary>code</summary>

| File                                                                                                       | Summary                                                                                                                                                                                                                                                                                                                                  |
| ---                                                                                                        | ---                                                                                                                                                                                                                                                                                                                                      |
| [utilities.ipynb](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/utilities.ipynb)   | Code snippet in code/utilities.ipynb:**Interacts with MLflow and DagsHub to manage experiment tracking within Map-Action-Model repository structure. Handles data sources and enables dataset manipulations.                                                                                                                             |
| [TFLearning.ipynb](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/TFLearning.ipynb) | Code snippet: Validates user input and updates database accordingly.Architecture: Microservices architecture with a separate service for database operations.Role: Ensures data integrity and security in the system.Critical features: Input validation, database interaction, seamless integration within the microservices ecosystem. |
| [zenml_running.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/zenml_running.py) | zenml_running.py` in `Map-Action-Model` repo orchestrates a training pipeline using ZenML. Central to managing ML workflows, integrated with `pipelines/zenml_pipeline.py`.                                                                                                                                                              |

</details>

<details closed><summary>code..zen</summary>

| File                                                                                                  | Summary                                                                                                                                                                          |
| ---                                                                                                   | ---                                                                                                                                                                              |
| [config.yaml](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/.zen/config.yaml) | Code in `code/.zen/config.yaml` sets active stack and workspace IDs for the repository. Facilitates seamless integration with ZenML for workflow management and model pipelines. |

</details>

<details closed><summary>code.steps.model</summary>

| File                                                                                                           | Summary                                                                                                                                                                                                                                                       |
| ---                                                                                                            | ---                                                                                                                                                                                                                                                           |
| [m_a_model.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/model/m_a_model.py) | Code snippet in `m_a_model.py` creates a modified VGG16 model for a specific class count. It adjusts the classifier and uses CrossEntropyLoss. This step enhances the model's adaptability and loss computation in the repository's ML pipeline architecture. |

</details>

<details closed><summary>code.steps.plot_metrics</summary>

| File                                                                                                                        | Summary                                                                                                                                                                              |
| ---                                                                                                                         | ---                                                                                                                                                                                  |
| [plot_metrics.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/plot_metrics/plot_metrics.py) | Code Summary:**`plot_metrics.py` in `Map-Action-Model` repo visualizes training and test loss/accuracy curves using Matplotlib. Enhances model evaluation insights for ML pipelines. |

</details>

<details closed><summary>code.steps.dagshub_utils</summary>

| File                                                                                                                                   | Summary                                                                                                                                                                                               |
| ---                                                                                                                                    | ---                                                                                                                                                                                                   |
| [dagshub_data_load.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/dagshub_utils/dagshub_data_load.py) | Code snippet in **dagshub_data_load.py** downloads and organizes data from a CSV file and DagsHub repository for machine learning model training in the **Map-Action-Model** repository architecture. |

</details>

<details closed><summary>code.steps.model_eval</summary>

| File                                                                                                                  | Summary                                                                                                                                                                                                                         |
| ---                                                                                                                   | ---                                                                                                                                                                                                                             |
| [evaluation.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/model_eval/evaluation.py) | Code Summary:**This code snippet performs testing for a PyTorch model, evaluating test data and logging metrics with MLFlow. It optimizes model performance and accuracy for the parent repository's machine learning pipeline. |

</details>

<details closed><summary>code.steps.training_step</summary>

| File                                                                                                                           | Summary                                                                                                                                                                                        |
| ---                                                                                                                            | ---                                                                                                                                                                                            |
| [training_step.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/training_step/training_step.py) | Code snippet in `training_step.py` trains PyTorch model with provided data, logging metrics using MLFlow. Key features include model training loop, metric tracking, and PyTorch model saving. |

</details>

<details closed><summary>code.steps.data_preprocess</summary>

| File                                                                                                                                             | Summary                                                                                                                                                                                                                                         |
| ---                                                                                                                                              | ---                                                                                                                                                                                                                                             |
| [data_loading_pipeline.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/data_preprocess/data_loading_pipeline.py) | Code Summary**:`data_loading_pipeline.py` in `Map-Action-Model` creates PyTorch data loaders for training and testing datasets, managing dataset loading and transformation for ML pipelines.                                                   |
| [data_transform.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/steps/data_preprocess/data_transform.py)               | Role:** Code snippet in `data_transform.py` for image preprocessing in `Map-Action-Model` repo architecture.**Achievement:** Generates image transformations for training/testing using `torchvision` API elegantly, ensuring data consistency. |

</details>

<details closed><summary>code.pipelines</summary>

| File                                                                                                                   | Summary                                                                                                                                                                                                                                                                       |
| ---                                                                                                                    | ---                                                                                                                                                                                                                                                                           |
| [zenml_pipeline.py](https://github.com/223MapAction/Map-Action-Model.git/blob/master/code/pipelines/zenml_pipeline.py) | Code snippet in `zenml_pipeline.py` orchestrates a machine learning training pipeline. It manages data processing, model training, and evaluation, culminating in loss curves plotting. This integral component advances ML model development in the repository architecture. |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `Python 3.x`

###  Installation

1. Clone the Map-Action-Model repository:

```sh
git clone https://github.com/223MapAction/Map-Action-Model.git
```

2. Change to the project directory:

```sh
cd Map-Action-Model
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running Map-Action-Model

Use the following command to run Map-Action-Model:

```sh
python main.py
```

###  Tests

To execute tests, run:

```sh
pytest
```
---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

See our [Contribution Guidelines](https://github.com/223MapAction/.github/blob/main/CONTRIBUTING.md) for details on how to contribute.


---

## 📄 License

This project is protected under the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/) License.

---

## Code of conduct

See our [Code of Conduct](https://github.com/223MapAction/.github/blob/main/CODE_OF_CONDUCT.md) for details on expected behavior in our community.

---

## 👏 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-quick-links)

---
