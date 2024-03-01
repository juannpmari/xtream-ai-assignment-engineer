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

## Architectural notes

#### Challenge 1
#### Challenge 2
The automated training pipeline is composed of a preprocessing step and a training step.
The preprocessing cleans and prepare the data through:
* data validation: checks for ilogical values (such as negative dimensions or price), and discards these rows
* data encoding: in order to leverage categorical features, they are encoded into numerical values using ordinal enconding. This can be done due to the features having ordinal order.
* data scaling: it's not necessary for regression trees, but it is for linear regression
* data splitting: divide the dataset into train and test splits.
The output of this step is the splitted dataset, which is saved to a local directory. Ideally, it could be saved to a cloud storage and versioned with DVC or similar tools.
The training step loads the latest preprocessed data, trains a model and evaluates it. The model weights are exported in .pt format for deployment, and are saved locally that simulates a model registry. Ideally, they would be saved to a real model registry with functionality for model versioning.
Metrics are saved to a .txt file. They should be tracked with experiment tracking tools, such as WandB.

For the sake of the exercise, the pipeline is scheduled for execution with a fixed frecuency. It checks for new data and, if found, runs and returns a new, fresh model. In a more complex scenario, with new data being continously stored in a data lake, a performance-based trigger could be set, so that the retraining launches whenever a certain metric goes below a threshold. This setup requires an according monitoring infrastructure.

The pipeline is executed inside a Docker container. Another consideration is that instead of having all the pipeline inside one container, it could be splitted in several microservices to gain more flexibility and maintanibility, but this would imply the use of an orchestration tool, such as Airflow.

#### Challenge 3
#### Challenge 4