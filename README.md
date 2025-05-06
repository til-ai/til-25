# DSTA BrainHack TIL-AI 2025

![Banner for TIL-AI](https://static.wixstatic.com/media/b03c31_bdb8962d37364d7c8cc3e6ae234bb172~mv2.png/v1/crop/x_0,y_1,w_3392,h_1453/fill/w_3310,h_1418,al_c,q_95,usm_0.66_1.00_0.01,enc_avif,quality_auto/Brainhack%20KV_v12_FOR_WEB.png)

**Contents**
- [DSTA BrainHack TIL-AI 2025](#dsta-brainhack-til-ai-2025)
  - [Get started](#get-started)
  - [Understanding this repo](#understanding-this-repo)
  - [Build, test, and submit](#build-test-and-submit)
  - [Links](#links)

## Get started

Here's a quick overview of the initial setup instructions. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

Use this repository as a template to create your own, and clone it into your Vertex AI workbench. You'll want to keep your repository private, so you'll need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

You'll also need to initialize the Git submodules:

```bash
git submodule update --init
```

Finally, create a Python virtual environment and install the development dependencies.

```bash
pip install -r requirements-dev.txt
```

## Understanding this repo

There's a subdirectory for each challenge: [`asr/`](/asr), [`cv/`](/cv), [`ocr/`](/ocr/), and [`rl/`](/rl). Each contains:

* A `src/` directory, where your code lives.
  * `*_manager.py`, which manages your model. This is where your inference and computation takes place.
  * `*_server.py`, which runs a local web server that talks to the rest of the competition infrastructure.
* `Dockerfile`, which is used to build your Docker image for each model.
* `requirements.txt`, which lists the dependencies you need to have bundled into your Docker image.
* `README.md`, which contains specifications for the format of each challenge.

You'll also find a final subdirectory, [`test/`](/test). This contains tools to test and score your model locally.

There are also two Git submodules, `til-25-finals` and `til-25-environment`. `finals` contains code that will be pulled into your repo for Semifinals and Finals. `environment` contains the `til_environment` package, which will help you train and test your RL model, and is installed by pip during setup. Don't delete or modify the contents of `til-25-finals/`, `til-25-environment/`, or `.gitmodules`.

## Build, test, and submit

Submitting your model for evaluation is simple: just build your Docker image, test it, and submit. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

You'll first want to `cd` into the directory you want to build. Then, build the image using Docker, following the naming scheme below. You should then run and test your model before using `til submit` to submit your image for evaluation.

```bash

# cd into the directory. For example, `cd ./asr/`
cd CHALLENGE

# Build your image. Remember the . at the end.
docker build -t TEAM_ID-CHALLENGE:TAG .

# Optionally, you can run your model locally to test it.
docker run -p PORT --gpus all -d IMAGE_NAME:TAG
python test/test_CHALLENGE.py

# Push it for submission
til submit TEAM_ID-CHALLENGE:TAG
```

## Links

* Your [Vertex AI Workbench on Google Cloud Platform](https://console.cloud.google.com/vertex-ai/workbench/instances?project=til-ai-2025) is where you'll do most of your development.
* The repo [Wiki]() (TK LINK) contains deep dives into setup guides, submission instructions, technical details, FAQs, and more.
* The [Guardian's Handbook](https://www.notion.so/tribegroup/BrainHack-2024-TIL-AI-Guardian-s-Handbook-c5d4ec3c3bd04b0db0329884c220791f), where you can find the Leaderboard and info about the competition.
* [Educational Content on Google Drive](https://drive.google.com/drive/folders/1JmeEwQZoqobPmUeSZWrvR5Inrw6NJ8Kr) from TIL-AI, which teaches you the basics of AI.
* The [#hackoverflow]() (TK LINK) channel on the TIL-AI Discord channel, a forum just for Guardians like you.

---

Code in this repo is licensed under the MIT License.
