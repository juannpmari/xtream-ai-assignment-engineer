# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create another Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run
Please fill this section as part of the assignment.

#### Challenge 1
It's addressed in `challenge1_notebook.ipynb`. It can be run using any code editor, and it will install the necessary dependencies. It could be run inside a virtual environment to avoid version conflicts if any packages are alredy installed:
* `python -m venv venv`

#### Challenges 2 and 3
The code corresponding to challenges 2 can be found in `./training_pipeline`, while that for challenge 3 is in `./restful-api`. All the code was containerized using Docker to make it simple to run for the final user, each in a separate container, and both managed by `docker-compose.yaml`. To run it, follow these steps:
* Build both docker images: each of the mentioned directories has a dockerfile. It may take a while due to the packages being installed
  * inside `./training_pipeline`, run `docker build -t training_pipeline:latest .`
  * inside `./restful-api`, run `docker build -t model-serving:latest .`
* Once both images are built, run `docker-compose up -d` to start the containers.
* Important bind mounts:
  * `./datasets` is the directory where raw .csv files are stored.
  * `./model_registry` is where the output model weights are stored, simulating a model registry.
  * `./metrics` is where training metrics (MAE) are stored for evaluation.
  * `config.json` is a config file with some important parameters explained below.
* Content of the config file `config.json`: it contains 3 main sections, for each step of the ML lifecycle
  * `preprocessing`:
    * `data_path`:`/datasets/diamonds` -> path to raw data.
    * `output_path`:`/datasets/diamonds/processed_data`, -> path where preprocessed data is stored.
    * ```categorical_variables:{
            "cut":[["Ideal", "Premium", "Very Good", "Good", "Fair"]],
            "color":[["D","E","F","G","H","I","J"]],
            "clarity":[["IF","VVS1", "VVS2","VS1","VS2","SI1","SI2","I1"]]
        }``` -> it encodes the categories and order of each categorical variable (cut, color and clarity).
  * `training`:
    * `model_output_path`:`/model_registry`, -> where the model weights are stored (path inside the container).
    * `metrics_output_path`:`/metrics`, -> where the training metrics are stored (path inside the container).
    * ```reg_tree_hyperparameters:{ 
            min_samples_split: 10, 
            min_samples_leaf: 4
        }``` -> hyperparameters for the regression tree, as calculated during hyperparameter tuning in challenge 1.
  * `deployment`: during the preprocessing step in model training, training data is used to fit a StandardScaler. So at deployment time, the exact same parameters must be used for the scaler, to ensure coherence between training and inference pipelines
    * `scaler_mean:[0.79411706,  1.09941872,  2.60753658 , 3.94086991, 61.71084386, 57.44624173,5.72623171,  5.72882141,  3.53367809]`
    * `scaler_std:[0.4679341 , 1.12201036 ,1.69109518 ,1.63173802 ,1.44541793 ,2.2595257,1.11614527, 1.10905561, 0.68836804]`
    
## Code details and extra notes

#### Challenge 2
The automated training pipeline consists of preprocessing, training and evaluation steps.

The preprocessing cleans and prepare the data through:
* data validation: checks for ilogical values (such as negative dimensions or price), and discards these rows.
* data encoding: in order to leverage categorical features, they are encoded into numerical values using ordinal enconding. This can be done due to the features having ordinal order.
* data scaling: considering that each feature has a different scale and range of values, it's convenient to scale them to make them more comparable to each other. While min-max scaling is very popular, it's more sensitive to outliers, so z-score scaling was the chosen method, which consists of transforming each feature so that it has mean=0 and std=1.
* data splitting: divide the dataset into train and test splits. The output of this step is the splitted dataset, which is saved to a local directory. Ideally, it could be saved to a cloud storage and versioned with DVC or similar tools.

The training step loads the latest preprocessed data, trains a model and evaluates it. As concluded in the challenge 1 notebook, a regression tree was chosen, though the pipeline architecture was designed to very easily allow for other models to be trained. The model weights are exported in .pt format for deployment, and are saved locally to a directory that simulates a model registry. Ideally, they would be saved to a real model registry with functionality for model versioning. Metrics are saved to a .txt file. They could be tracked with experiment tracking tools, such as WandB.

For the sake of the exercise, the pipeline is scheduled for execution with a fixed frecuency (for this example, every 50 minutes), which it should be stablished based on how often new data becomes available. With every execution, the code checks if there's new data available (assuming the .csv are named with their creation date) and, if found, runs and returns a new, fresh model, along with it's evaluation metrics. In a more complex scenario, with new data being continously stored in a data lake, a performance-based trigger could be set, so that the retraining launches whenever a certain metric goes below a threshold. This setup requires an according monitoring infrastructure.
The pipeline code expects each new .csv to be named like `diamonds_2024-03-07-04-17-03.csv`.

The pipeline is executed inside a unique Docker container. In order to to gain more flexibility and maintanibility, instead of having all the pipeline inside one container, it could be splitted in several independent microservices, but this would require the use of an orchestration tool, such as Airflow.

After training, a validation step could be set for continuous deployment, which verifies that the new models performs better than the current one. Once deployed, there are many online testing strategies available to minimize the risk of the new model performing badly in production.

#### Challenge 3
A RESTful API was created using FastAPI. It loads the latest model from the local directory `model_registry`, and deploys it for inference using PyTorch. In can be accesed on `http://localhost:8000/docs`
It exposes one endpoint `/predict` that can be used both for real-time prediction (one sample at a time) or for batch prediction. AS input, tt expects a list of dictionaries, each with the raw features of a diamond:
```
[
  {
    "carat": float,
    "cut": str,
    "color": str,
    "clarity": str,
    "depth": float,
    "table": float,
    "x": float,
    "y": float,
    "z": float
  },
  {
    ...
  }
]
```

It performs the same preprocessing steps defined for training pipeline, and returns a dict with predictes prices as a list:
```
{
  "msg": "Diamond price predicted succesfully",
  "pred_prices": "[530.0, 17329.0]"
}
```
An important detail is that,as explained above, we need to manually add the mean and standard deviation for the scaler to the config file, which are obtained during model training, to ensure that the exact same transformations are applied during training and deployment. In the future, these values should be saved automatically during the execution of the training pipeline.
In case of running the server manually (it shouldn't be necessary as the dockerfile runs it automatically), you need to run `uvicorn app:app --host 0.0.0.0 --port 8000`.

#### Challenge 4
Please refer to `challenge4.pdf`.