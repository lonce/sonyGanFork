# sonyGanForked

 
 This repo is a fork of [Comparing-Representations-for-Audio-Synthesis-using-GANs](https://github.com/SonyCSLParis/Comparing-Representations-for-Audio-Synthesis-using-GANs).  The requirements and install procedure are updated a bit from the original so that it works and is sharable through containerization. 
 
 The Docker version assumes you are running on nvidia graphics cards and have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)  installed.
 
 The Singularity version is for internal use (we use an image built on our university computers that run on a rack of V100s), but it is built with same libraries and packages as found in the requirements.txt and docker file. 



# The dataset
You can run some test using the same dataset used by the Sony folks:  the [Nsynth datasaet](https://magenta.tensorflow.org/datasets/nsynth). Scroll down to 'Files' and grab the json/wav version of the 'Train' set (A training set with 289,205 examples). It's big, something approaching 20Gb.

___
___

# <span style="color:maroon"> DOCKER </span>
This section is for running in an nvidia-docker container. For running in a Singularity container (used on NUS atlas machines), see <span style="color:maroon"> SINGULARITY </span> below.
## <span style="color:maroon"> Install </span>
0) install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1) Build the image foo with tag bar:
```
   $ cd docker
   $ docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile --tag foo:bar ../
```

## <span style="color:maroon"> Running </span>
The first thing you need to do to use this code is fire up a container from the image *from the sonyGanForked directory* (don't change the first -v mounting arg). The second mounting arg needs the full path to your data directory before the colon, leave the name 'mydata' as it is. 'foo' and 'bar' are whatever name and tag you gave to your docker image.
```
 $ docker run  --shm-size=10g --gpus "device=0" -it -v $(pwd):/sonyGan -v /full/path/to/datadir:/mydata --rm foo:bar
```
(You'll see a few warnings about depcrated calls that seem to be harmless for the time being). 

## <span style="color:maroon"> Smallish test run </span>
This runs a small configuration, creating output in output/outname:
```
runscripts/runtrainTEST.sh  -n outname config_files/new/testconfig.json
```
You will see that the script ouputs some text, the last line gives you the command to run to watch the stderr output flow (tail -f logsdir.xxxx/stderr.txt). Copy and paste it to the shell. 
Depending on your machine, it could take 20 minutes to run, but that is long enough to then generate a discernable, if noisy musical scale from nsynth data. 

##  <span style="color:maroon"> Training a new model </span>
Now your can train by executing a script:
```
runscripts/runtrain.sh -n outname  <path-to-configuration-file>
```
-n outname overrides the "name" field in the config file. 

___
___


# <span style="color:green"> SINGULARITY </span>
This section is for running in a Singularity container. For running in a Docker container (e.g. on your local machine), see <span style="color:maroon"> DOCKER </span> below.

## <span style="color:green">  Install </span>
0) Have an account on NUS high-performance computing. Run on the atlas machines. 
1) There is already an image built with all the libs and python packages we need. You can assign it to an environment variable in a bash shellor bash script:
```
   $ image=/app1/common/singularity-img/3.0.0/user_img/freesound-gpst-pytorch_1.7_ngc-20.12-py3.simg
```


## <span style="color:green"> Running </span>
### <span style="color:green"> running interactively </span>
I think you have to be logged in on atlas9.nus.edu.sg. This example runs the smallish test run on nsynth using mel and just a few iterations per progressive gan scale.
```
 $ qsub -I -l select=1:mem=50GB:ncpus=10:ngpus=1 -l walltime=01:00:00 -q volta_login
 # WAIT FOR INTERACTIVE JOB TO START, then create container:
 $ singularity exec $image bash
 # Now you can run your python code:
 $ python train.py --restart  -n "mytestoutdir" -c $configfile -s 500 -l 200
```
(You'll have to set the ouput_path, the data_path, and the att_dict_path values in the config file first. The -n arg is the name of the folder inside the output_path where your checkpoints will be written). 

### <span style="color:green"> Submitting non-interactive jobs </span>
```
 $ qsub runscripts/runtrain.pb
 # Or for running the small test non-interactively:
 $ qsub runscripts/runtrainTEST.pb
```
Unfortuneately, you can't pass args to a scripit that you are submitting with qsub. Thus you will need to edit these scripts to set the output folder name and the config file you want to use.  You will also have to set the output_path, the data_path, and the att_dict_path values in the config file.

You'll notice that I use rsynch to move the data to the /scratch directory on the machine the system allocates to run the job (you don't have control over which machine that is). I normally do this to speed up the I/O between the GPU and the disk, but for sonyGAN, the preprocessing step writes the preprocessed data to the output_path which is probably on your local disk anyway, subverting any attempts to speed things up this way. 

## <span style="color:green">  Viewing the error plots after a run  </span>
You can run a notebook on atlas9 while interacting with it through a browser on your local machine. :
```
 $ qsub -I -l select=1:mem=50GB:ncpus=10:ngpus=1 -l walltime=01:00:00 -q volta_login
 # WAIT FOR INTERACTIVE JOB TO START, then create container running jupyter and exposing a port:
 $ singularity exec $image jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
 #Then, BACK ON YOUR LOCAL MACHINE:
 $ ssh -L 8888:volta01:8889 user_name@atlas9
 # Then just point your browser to: http://localhost:8888
```

___
___



# Plotting the output
Run jupyter notebook, and open plotPickle.ipynb. In the notebook, set the 'infile' to be the xxx_losses.pkl file in your output directory. Then just run all the cells in the notebook. 
 

# Example of config file:
The experiments are defined in a configuration file with JSON format.
```
# This is the config file for the 'best' nsynth run in the Sony Gan paper (as far as I can tell). 
{
    "name": "myTestOut", #outputfolder in ouput_path for checkpoints, and generation. **THIS SHOULD BE CHANGED FOR EVERY RUN YOU DO!!**  (unless you want to start your run from the latest checkpoint here). This field should actually be provided by a flag - not stuck in a configuration file!
    "comments": "fixed alpha_configuration",
    "output_path": "output",  # Used for preprocessed data and checkpoints. These files can be **BIG** - make sure you have space whereever you put it. On NUS machines, consider using /hpctmp/user_name/foo...
    "loaderConfig": {
        "data_path": "/mydata/nsynth-train/audio", # Path to audio data. ('mydata' matches the mount point in the 'docker run' command above.)
        "att_dict_path": "/mydata/nsynth-train/examples.json", # Path to meta data file
        "filter": ["acoustic"],
        "instrument_labels": ["brass", "flute", "guitar", "keyboard", "mallet"],
        "shuffle": false,

        # I *think* this is just used to filter the files used for training
        "attribute_list": ["pitch"], #the 'conditioning' param used for the GAN
        "pitch_range": [44, 70], #the full range of pitches in the nsynth db
        "load_metadata": true,

        # not sure why this is here. Instances that match filter critera are counted in code
        "size": 24521   #the number of training examples(???)
    },
        
    "transformConfig": {
		"transform": "specgrams", #the 'best' performer from the sony paper
        "fade_out": true,
        "fft_size": 1024,
        "win_size": 1024,
        # I think n_frames and hop_size  are essentially specifying how long the wave files are. 
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
        "imagefolderDataset": true, # ?? what does this param do?
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

        # from original AC-GAN paper nad Nistal paper
        "weightConditionG": 10.0, #in AC-GAN, weight of the classification loss applied to the generator
        "weightConditionD": 10.0, #in AC-GAN, weight of the classification loss applied to the discriminator
        # used for conditional training
        "attribKeysOrder": {
            "pitch": 0
        },
        "startScale": 0,
        "skipAttDfake": []
    }
}

```


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

Note: you have to be in a container (Docker or Singularity) to generate.
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

# Key background papers
[Odena's AC-GAN](https://arxiv.org/pdf/1610.09585.pdf)  
[Wasserstein earth-mover distance] (https://arxiv.org/pdf/1704.00028.pdf)  
[GANSynth] (https://arxiv.org/abs/1902.08710?)  

## Notes
1) We are not seeing the kind of convergence in the error measures we expected from the nsynth results reported in the Comparing Representation paper. Several errors grow, particularly in the last prgressive stage of the GAN. However, the sounds that the network generates after training are still of very high quality. Please drop us a note if you find parameters that make these error measures behave better.   

2)This repo is evolving. Please submit issues!

