# Code for the paper "Can LLMs Correct Physicians, Yet? Investigating Effective Interaction Methods in the Medical Domain"

## Step 1: Load the Harness Framework

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Step 2: 

Insert python and yaml files into `lm-eval/tasks/pubmedqa` directory

## Step 3: 

Open `lm_eval/tasks/__init__.py` and insert the following code snippet right after the imports of libraries:

```bash
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case1a import PubmedqaLongBinaryCase1a
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case1b import PubmedqaLongBinaryCase1b
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case1c import PubmedqaLongBinaryCase1c
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case1d import PubmedqaLongBinaryCase1d
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case2a import PubmedqaLongBinaryCase2a
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case2b import PubmedqaLongBinaryCase2b
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case2c import PubmedqaLongBinaryCase2c
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case2d import PubmedqaLongBinaryCase2d
from lm_eval.tasks.pubmedqa.pubmedqa_long_binary_case3 import PubmedqaLongBinaryCase3
```

## Step 4:

Add your Huggingface login credentials to the main function in `lm_eval/__main__.py`
It should be something like this:

```bash
if __name__ == "__main__":
    from huggingface_hub import login
    login("your_key_to_login_the_hugging_face")
    cli_evaluate()
```

## Step 5:

Add the path to your directory within .py and .sh files you want to run.
Make sure that you pick the scenario you want to test inside the corresponding python file.

## Step 6:

Run the code via .sh files. Output of the model will be saved in the directory you specified in the script and in the python file.

## Step 7:

To postprocess the model outputs, run `postprocess_case1.ipynb` or `postprocess_case2`
To postprocess the model outputs for Case 3, you can use `postprocess_case1.ipynb`

# Citation

If you use this code or our paper, please cite:

```bash
@article{sayin2024LLMsPhysicianInteraction,
      title={Can LLMs Correct Physicians, Yet? Investigating Effective Interaction Methods in the Medical Domain},
      author={Burcu Sayin and Pasquale Minervini and Jacopo Staiano and Andrea Passerini},
      year={2024},
      journal={arXiv},
      volume={abs/2403.20288}
}
```