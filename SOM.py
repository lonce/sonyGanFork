import argparse
import json
import os # to create output directory
import subprocess # to run shell command to make wave and param files from .pt file 

import torch

# import GAN utilities
from evaluation.gen_tests.generation_tests import StyleGEvaluationManager
from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, makesteps

# import mesh adaptation module
import SOM.SOM as som
#-------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generation testing script', add_help=False)
parser.add_argument('-c', '--config', help="config file with parameters",
                        type=str, dest="configfile")

args, unknown = parser.parse_known_args()
print(f'Will read params from {args.configfile}')
data = json.load(open(args.configfile,))

# preprocess data
for key in data :
	if data[key]=="True" : data[key]=True
	if data[key]=="False" : data[key]=False
	if data[key]=="None" : data[key]=None	
	if key=="z0" or key=="z1" or key=="z2" or key=="z3" : data[key]=torch.load(data["checkpointDir"]+data[key])
	if key=="outpath"  : data[key] = data["checkpointDir"]+data[key]
	if key=="outptfile"  : data[key] = data["outpath"]+"/"+data[key]


# create variables from data in config file
xlength = data["xlength"]
ylength = data["ylength"]
minPitch = data["minPitch"]
maxPitch = data["maxPitch"]
directions = data["directions"]
pitchDim = data["pitchDim"]
step = data["step"]
iterations = data["iterations"]
log_iterations = data["log_iterations"]
clampedges = data["clampedges"]
checkpointDir = data["checkpointDir"] 
outpath = data["outpath"]
outptfile = data["outptfile"]
z0 = data["z0"]
z1 = data["z1"]
z2 = data["z2"]
z3 = data["z3"]

#-------------------------------------------------------------------------------------

# Load the model
model, config, model_name = load_model_checkp(checkpointDir)
eval_manager = StyleGEvaluationManager(model, n_gen=2)

# Create the SOM instance
p2 = som.kmap(ylength,xlength,z0.cpu(),z1.cpu(),z2.cpu(),z3.cpu(), vizdims=eval_manager.latent_noise_dim) #( ylength (num rows), (xlength (num cols))
print(f'Mesh shape is: {p2.weights.shape}')

#-------------------------------------------------------------------------------------

# Here we set the one-hot part of the latent vector to be the values for requested pitches
# Eg. if pitchDim==1, set a single pitch for given column, incrementing across columns
# Have to do this *after* creating the kmap because that's where the grid of latents is generated. 

if (pitchDim == 0  and directions != [2,6]) or  (pitchDim == 1  and directions != [0,4]):
	print("WARNING: It appears that you will be adapting the mesh in the direction that the one-hot pitch vector changes. Probably not what you want to do.")

if minPitch != None :
    if pitchDim == 0 :
        numPitches=ylength 
    if pitchDim == 1 :
        numPitches=xlength
    if (maxPitch-minPitch) != 0  : #if not asking for constant pitch, check if number of pitches agress with number of mesh points
        assert abs(maxPitch-minPitch)+1 == numPitches, f"pitch range is [{minPitch},{maxPitch}], but dimension {pitchDim} is of length {numPitches}"

    for i in range(ylength) :
        for j in range(xlength) :
            if pitchDim == 0 : # increment pitch across rows
                p2.weights[i, j] = som.setPitch(p2.weights[i, j], eval_manager, "pitch", i+minPitch)
            elif pitchDim == 1 : # increment pitch across columns
                p2.weights[i, j] = som.setPitch(p2.weights[i, j], eval_manager, "pitch", j+minPitch)
            else : # all sounds at same pitch
                p2.weights[i, j] = som.setPitch(p2.weights[i, j], eval_manager, "pitch", minPitch)

#-------------------------------------------------------------------------------------
# get the spectrograms for the  mesh point latents
batched_latents=torch.Tensor(p2.weights.reshape(ylength*xlength,p2.weights.shape[2]))
gen_batch = model.test(batched_latents, toCPU=True, getAvG=True)

gen_batch=gen_batch.numpy().reshape(ylength,xlength,257,128)

#-------------------------------------------------------------------------------------

for i in range (iterations) : 
    # get the new spectrograms at the latent locations on the mesh
   
    # Stack up weights into a single list for batch processing bythe GAN
    batched_latents=torch.Tensor(p2.weights.reshape(ylength*xlength,p2.weights.shape[2]))
    
    # Compute the values of the functions at each node location 
    gen_batch = model.test(batched_latents, toCPU=True, getAvG=True)
    gen_batch_npmatrix=gen_batch.numpy().reshape(ylength,xlength,257,128)
    
    # move the latents for the mesh points, return values are just for logging
    changesum, diffm=p2.weightUpdate(gen_batch_npmatrix, step=step, clampedges=clampedges, directions=directions)

    if i%log_iterations == 0 : 
    	print(f"iteration {i}: changesum is {changesum}")


#-----------------------
# save pt file
if not os.path.exists(outpath):
    os.makedirs(outpath)

torch.save((gen_batch,batched_latents), outptfile)
print(f"Wrote {outptfile}")

# Now create the wave and param files for all the mesh points stored in the .pt file
subprocess.run(["python", "spectset2paramsound.py", "--gen_batch", str(outptfile), "--d0", str(ylength), "--d1", str(xlength), "--pm", "True", "-d", str(checkpointDir)])
print(f"Done. ")