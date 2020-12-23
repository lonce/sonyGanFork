# sonyGanForked
This repo is a fork of [Comparing-Representations-for-Audio-Synthesis-using-GANs](https://github.com/SonyCSLParis/Comparing-Representations-for-Audio-Synthesis-using-GANs).  The requirements and install procedure are different from the original so that it works and is sharable assuming you are running on nvidia graphics cards and have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)  installed.
# Install
0) install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1) Build the image foo with tag bar:
```
   $ cd docker
   $ docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile --tag foo:bar ../
```


# The dataset
You can run some test using the same dataset used by the Sony folks:  the [Nsynth datasaet](https://magenta.tensorflow.org/datasets/nsynth). Scroll down to 'Files' and grab the json/wav version of the 'Train' set (A training set with 289,205 examples). It's big, something approaching 20Gb.

# Running
The first thing you need to do to use this code is fire up a container from the image *from the sonyGanForked directory* (don't change the first -v mounting arg). The second mounting arg needs the full path to your data directory before the colon, leave the name 'mydata' as it is. 'foo' and 'bar' are whatever name and tag you gave to your docker image.
```
 $ docker run  --shm-size=10g --gpus "device=0" -it -v $(pwd):/sonyGan -v /full/path/to/datadir:/mydata --rm foo:bar
```
(You'll see a few warnings about depcrated calls that seem to be harmless for the time being). 

# Smallish test run
This runs a small configuration, creating output in output/outname:
```
runscripts/runtrainTEST.sh  -n outname config_files/new/testconfig.json
```
You will see that the script ouputs some text, the last line gives you the command to run to watch the stderr output flow (tail -f logsdir.xxxx/stderr.txt). Copy and paste it to the shell. 
Depending on your machine, it could take 20 minutes to run, but that is long enough to then generate a discernable, if noisy musical scale from nsynth data. 

# Training a new model
Now your can train by executing a script:
```
runscripts/runtrain.sh -n outname  <path-to-configuration-file>
```
-n outname overrides the "name" field in the config file. 
# Example of config file:
The experiments are defined in a configuration file with JSON format.
```
# This is the config file for the 'best' nsynth run in the Sony Gan paper (as far as I can tell). 
{
    "name": "myTestOut", #outputfolder in ouput_path for checkpoints, and generation. THIS SHOULD BE CHANGED FOR EVERY RUN YOU DO!!  (unless you want to start your run from the latest checkpoint here). This field should actually be provided by a flag - not stuck in a configuration file!
    "comments": "fixed alpha_configuration",
    "output_path": "output",
    "loaderConfig": {
        "data_path": "/mydata/nsynth-train/audio", #'mydata' matches the mount point in the 'docker run' command above
        "att_dict_path": "/mydata/nsynth-train/examples.json",
        "filter": ["acoustic"],
        "instrument_labels": ["brass", "flute", "guitar", "keyboard", "mallet"],
        "shuffle": false,
        "attribute_list": ["pitch"], #the 'conditioning' param used for the GAN
        "pitch_range": [44, 70], #the full range of pitches in the nsynth db
        "load_metadata": true,
        "size": 24521   #the number of training examples(???)
    },
        
    "transformConfig": {
		"transform": "specgrams", #the 'best' performer from the sony paper
        "fade_out": true,
        "fft_size": 1024,
        "win_size": 1024,
        "n_frames": 64,
        "hop_size": 256,
        "log": true,
        "ifreq": true,          # instantaneous frequency (??)
        "sample_rate": 16000,
        "audio_length": 16000
    },
    "modelConfig": {
        "formatLayerType": "gansynth",
        "ac_gan": true,
        "downSamplingFactor": [
            [16, 16],
            [8, 8],
            [4, 4],
            [2, 2],
            [1, 1]
        ],
        "imagefolderDataset": true,
        "maxIterAtScale": [200000, 200000, 200000, 300000, 300000 ],
        "alphaJumpMode": "linear",
        # alphaNJumps*alphaSizeJumps is the number of Iters it takes for alpha to go to zero. The product should typically be about half of the corresponding maxIterAtScale number above.
        "alphaNJumps": [3000, 3000, 3000, 3000, 3000],
        "alphaSizeJumps": [32, 32, 32, 32, 32],
        "transposed": false,
                "depthScales": [ 
            128,
            64,
            64,
            64,
            32
        ],
        "miniBatchSize": [12, 12, 12, 8, 8],
        "dimLatentVector": 64,
        "perChannelNormalization": true,
        "lossMode": "WGANGP",
        "lambdaGP": 10.0,
        "leakyness": 0.02,
        "miniBatchStdDev": true,
        "baseLearningRate": 0.0008,
        "dimOutput": 2,
        "weightConditionG": 10.0,
        "weightConditionD": 10.0,
        "attribKeysOrder": {
            "pitch": 0
        },
        "startScale": 0,
        "skipAttDfake": []
    }
}

```

# Plotting the output
I haven't figured out how sony visualizes the metrics of the training run. What I do is extract data from the logs that can then be read in a python notebook and plotted. OK for now:
```
cd logs
stderr2data.sh    <path-to-log-files>/stderr.txt > <path-to-log-files>/plotable.txt
```
Now you can run the jupyter notebook, plotLosses.ipynb, set the path and stages parameters in the cell labels '# Set parameters in this cell' and then run all cells to get your plots. 

# Evaluation 
### (from sony - I haven't tried this yet)
You can run the evaluation metrics described in the paper: Pitch Inception Score (PIS), Instrument Inception Score (IIS), Pitch Kernel Inception Distance (PKID), Instrument Kernel Inception Distance (PKID) and the [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) (FAD).

* For computing Inception Scores run:
```
python eval.py <pis or iis> --fake <path_to_fake_data> -d <output_path>
```

* For distance-like evaluation run:
```
python eval.py <pkid, ikid or fad> --real <path_to_real_data> --fake <path_to_fake_data> -d <output_path>
```

# Synthesizing audio with a model
```
python generate.py <random, scale, interpolation or from_midi> -d <path_to_model_root_folder>
```
(sony didn't include the code for generating audio from midi files). 
Output is written to a sub/sub/sub/sub/ folder of output_path.
scale -gerates a bunch of files generated using latent vectors with the conditioned pitch value set.  
interpolation - generates a bunch of files at a given pitch interpolating from one random (?) point in the latent space to another.   
random - generates a bunch of files from random points in the latent space letting you get an idea of the achievable variation. 


# Audio examples (from sony)
[Here](https://sites.google.com/view/audio-synthesis-with-gans/p%C3%A1gina-principal) you can listen to audios synthesized with models trained on a variety of audio representations, includeing the raw audio waveform and several time-frequency representations.
## Notes
This repo is still a work in progress. Please submit issues!
