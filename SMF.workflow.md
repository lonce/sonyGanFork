# Sound Model Factory workflow

 This document steps you through the SMF process documented in the NEURIPS'21 submission. 



<img src="/home/lonce/working/sonyGanFork/resources/system-schematic.png" width="500px">

## <span style="color:maroon"> 1. GAN Training </span>

1. Prepare a dataset
   1. a) Create a directory with your collection of audio files. You config file will point to this directory
2. Start Docker
   1. a) fire up a docker container from the image *from the sonyGanForked directory* (don't change the first -v mounting arg). The second mounting arg needs the full path to your data directory before the colon, leave the name 'mydata' as it is. 'foo' and 'bar' are whatever name and tag you gave to your docker image.

```
 $ docker run  --shm-size=10g --gpus "device=0" -it -v $(pwd):/sonyGan -v /full/path/to/datadir:/mydata --rm foo:bar
```
	3. Prepare you config file (see README.md) 
 	4. Run a training script. This runs a small configuration, creating output in output/outname:

```
runscripts/runtrain.sh -n outname  <path-to-configuration-file>
```
​	-n outname overrides the "name" field in the config file. 

## <span style="color:maroon"> 2. Create a Grid </span>

1. (Still in a container). Here we generate random samples from the GAN that can be auditioned for selection as corners of the subspace you will define for the synthesizer training.

```
python generate.py random -d <path_to_model_root_folder>
```
Output is written to a sub/sub/sub/sub/ folder of output_path, and files are in pairs, .wav and .pt files (the pt files store the latent parameters for the sounds).

2. Choose 4 files as the corners of your latent subspace that your final synth will traverse. Pass the .pt files for those 4 points as the z0-3 args to this command to fill out the grid with sound samples: 

```
python generate.py 2D4pt --z0 path/filename0.pt --z1 path/filename1.pt --z2 path/filename2.pt --z3 path/filename3.pt --d0 [number of grid pts in z0-z1 dimension] --d1 [number of grid pts in z0-z2 dimension]  --pm True -d outputdir
```

​	--pm True means same param files in paramManager format (used by the MTCRNN for conditional training)  

## <span style="color:maroon"> 2a. Interact! </span>

1. Whenever you have a grid of sound samples, you can use the hand-dandy browser app to display and play sounds. Clone: https://github.com/lonce/svgSoundGrid , put the sounds in the resources directory, and change the load function to load your files. 

## <span style="color:maroon"> 3. Linearize with the SOM </span>

1. You need to run the SOM notebook in a docker (so that it can call the GAN to generate sound samples), with port mapping to that you can access notebook webapge in a browser running on your machine (not in a docker). So start your docker like this:

   ```
   docker run  --ipc=host --gpus "device=1" -it -v $(pwd):/sonyGan -v /scratch/lonce:/mydata -p 5555:8888 lonce:tifresi
   ```

2. Run jupyter notebook SOM.ipynb in the container.

3. Point your browser to the notebook web page.

   To get the ip address, in another window (not the container), run:

   ```
   docker inspect container_name
   ```

   and find the IPV4 address of the container.  Point a browser to that address with the port you mapped (or use the URL provided from running jupyter). 

4. Edit the netbook cell following **Set your PARAMs here** to point to the 4 corner .pt files, etc. 

5. Run all cells (note the first part of the notebook is just a demo of the SOM)

6. Save your adapted grid as a .pt file (see cell **SAVE the grid**)

7.  In your docker, run: 

   ```
   python spectset2paramsound.py --gen_batch linearized.pt --d0 21 --d1 21 --pm True -d output/oreilly2
   ```

   to generate the wave files and paramManage files for training the RNN

8. (You can explore the new grid using the svgSoundGrid described in 2a above, as well. )

   

## <span style="color:maroon"> 4. Train the RNN </span>



1.  The RNN is a separate GitHub project: https://github.com/lonce/MTCRNN.Fork

2.  There is a docker file for creating a ready-to-run container.

3. There is a README.md in the project directory, but you can see **train.template.sh** for an example of a "typical" runscript for training. The script takes no arguments from the command line. 

4. GENERATION

   1. First use the MakeConditioningArray.ipynb to create a parameter file for controlling the conditioned parameters over time. I usually generate 1000 time points for a nominal 1 second, and then in the generate.sh script, scale that to the desired duration of the generated wave file. There are quite a few example in the notebook for generating parameter files.  

      The output is a .py file with an n-dimensional array. The parameters are not named in the file, so must be in the same order as they are listed in the generate.sh script. 

      (If you want to generate LOTS of files, each with different constant parmeter values, see the companion notebook, MakeConditioningParamFiles.ipynb)

   2.  The customize **runscripts/generate.template.sh** - you'll need to edit this script to set your directories and the parameters to match those you trained with. The run it. 

      

   

   ## 