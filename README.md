# this-is-a-anonymous-repo-from-me



# Multitask Edit Project

This project implements a novel multitask edit approach using the `Transformers` and `PEFT` libraries, tailored to predict and perform changes to Stack Overflow (SO) posts by considering their different components. While initially developed for enhancing content in community-driven Q&A platforms like Stack Overflow, this framework is versatile enough to be adapted for a variety of similar tasks in other domains. It offers flexible configurations for model parameters and prompt tuning settings, making it an ideal choice for researchers and practitioners seeking to optimize performance across diverse applications, especially those involving nuanced content modification and updates.



## Environment Setup

The project requires a Python environment with specific libraries. We use Python 3.11 and CUDA 11.7.0 for development, ensuring compatibility and performance optimization. Follow these steps to set up your environment:

### Step 1:  Create a Python Virtual Environment

Create a virtual environment using `virtualenv`:

```
python -m venv /path/to/your/venv
```

Activate the virtual environment:

```
source /path/to/your/venv/bin/activate
```

### Step 2:  Install Dependencies

Install the required Python libraries:
```
pip install -r requirements.txt
```

### Running 
Before running the training script, you can adjust the model configuration by modifying parameters in the `run.sh`.

## Modifying Hyperparameters
Open the `run.sh`file, and you will see several configurable parameters:
**MODEL_NAME**: The model name, default is `"google/flan-t5-base"`.
**BATCH_SIZE**: Batch size, default is `8`. 
**LEARNING_RATE**: Learning rate, default is `1e-4`.
**TOTAL_EPOCHS**: Number of training epochs, default is `5`.
You can modify these parameters as needed.

## Training
Execute the following command to start training:
```
./run.sh
```

After running the script, the results of the training will be saved in the specified output directory. By default, models and training metrics (like loss and evaluation metrics) will be saved in the `checkpoints_source` directory.

To initiate the second part of the training process, modify the data loaders in the script as follows:
```
train = DataLoader(MyDataset("train", "target"), shuffle=True,  batch_size=args.batch_size, collate_fn=collate_fn)
val = DataLoader(MyDataset("val", "target"), shuffle=False,  batch_size=args.batch_size, collate_fn=collate_fn)
```
This adjustment switches the dataset mode to "target", allowing the model to be trained on a different set of data or tasks, as configured in your setup.


## Notes

Ensure your machine has CUDA 11.7.0 installed and a CUDA-compatible version of PyTorch to utilize GPU acceleration for training. Additionally, this project is designed to support deployment on local machines, specifically those equipped with an NVIDIA GeForce RTX 4090 with 24 GB of memory. This configuration allows for optimal performance leveraging local hardware capabilities.


