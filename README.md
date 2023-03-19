# SLNSpeech
SLNSpeech: solving extended speech separation problem with the help of sign language![model](C:\Users\lx\Desktop\1\SLNSpeech\imgs\model.png)

# Dataset

## SLNSpeech Dataset

Due to copyright issues, we cannot directly disclose our dataset, but we will publish the features extracted using our model and some samples later.

## Prepare data

We hope the data to be categorized by person, that is, the videos of a certain speaker are in the same directory. First, using ffmpeg to cut the videos into video frames and audios, and storing the audios into audio directory and storing the video frames into frames directory, still categorized by person.  The directory structure is as follows.

```
-dataset
	-audio
		-speaker1
		-speaker2
		-speaker3
		...
	-frames
		-speaker1
		-speaker2
		-speaker3
		...
```

Second, through `python create data/create_ Index.py` creates a csv file contaning addresses which store speaker visual frames and audio. In `data/create_ In index.py`, it is necessary to set the gender information of speakers.

# Training

```
python main.py --list_train 'path/train.csv' --list_test 'path/test.csv'
```

All parameters are included in `arguments.py` and can be changed according to demand.
