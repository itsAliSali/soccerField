# soccerField
In this project, I detected the players and classified them using background subtraction methods and artificial neural networks. The primary libraries that I used are openCV and PyTorch.

# summary
In part1.py, the only functionality is reading the video from local storage and detect players.
In part2_classify.py, the classification is added on top of part1.py.
Some scripts were developed to generate the dataset and train the neural net.

# input
The input videos are from [Soccer video and player position dataset](https://dl.acm.org/doi/10.1145/2557642.2563677). I used about 4 minutes of the first half to train my neural network. And a video from the second half was used to evaluate the model. Accuracy scores were about 97%. The codes are optimized to run on GPU.
<br>
<img src="https://github.com/itsAliSali/soccerField/blob/b4a6f703d806a3f7eab033e5aa3ef7cbb3189a08/figures/input.png" width="550" height="400">

# output
Here is an image of top view of the field with a rendered map of players position:
<br>
<img src="https://github.com/itsAliSali/soccerField/blob/90423d95d6fad78ec0b7c128439ff070368d17b2/figures/outputs.png" width="650" height="500">

# dataset
I automatically generated the patches of players and labeled them by checking the color by openCV. Here is a figure containing 40 patches (from test dataset) and label/prediction as the title:
<br>
<img src="https://github.com/itsAliSali/soccerField/blob/b4a6f703d806a3f7eab033e5aa3ef7cbb3189a08/figures/dataset.png" width="600" height="500">

