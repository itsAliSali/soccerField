# soccerField
In this project, I detected the players and classified them using background subtraction methods and artificial neural networks. The primary libraries that I used are openCV and PyTorch.

# summary
In part1.py, the only functionality is reading the video from local storage and detect players.
In part2_classify.py, the classification is added on top of part1.py.
Some scripts were developed to generate the dataset and train the neural net.

# input
The input videos are from [Soccer video and player position dataset](https://dl.acm.org/doi/10.1145/2557642.2563677). I used about 4 minutes of the first half to train my neural network. And a video from the second half was used to evaluate the model. Accuracy scores were about 97%. The codes are optimized to run on GPU.
<br>
<img src="https://github.com/itsAliSali/" width="640" height="400">

# output
Here is an image of top view of the field with a rendered map of players position:
<br>
<img src="https://github.com/itsAliSali/" width="500" height="500">

# dataset
I automatically generated the patches of players and labeled them by checking the color by openCV. Here is a figure containing 40 patches (from test dataset) and label/prediction in the title:
<br>
<img src="https://github.com/itsAliSali/" width="500" height="500">

