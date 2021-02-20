import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, makesteps
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioPreprocessor

import numpy as np

from librosa.output import write_wav

import soundfile as sf

##-----------   paramManager  interface   ------------------##
from paramManager import paramManager


def generate(parser):
    args = parser.parse_args()

    argsObj=vars(args)
    print(f"generate args: {argsObj}")

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    postprocess = AudioPreprocessor(**config['transformConfig']).get_postprocessor()
    #### I WANT TO ADD NORMALIZATION HERE  ######################
    print(f"postprocess: {postprocess}")

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "2D")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))




    
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=2)

    z0=torch.load(argsObj["z0"])
    z1=torch.load(argsObj["z1"])

    minpitch=int(argsObj["p0"])
    maxpitch=int(argsObj["p1"])
    pitchrange=maxpitch-minpitch
    if pitchrange < 1 : 
        pitchrange=1

    interp_steps1=int(argsObj["d1"])
    interp_steps1norm=interp_steps1 -1 # because the batch generater will spread the steps out to include both endpoints

    usePM=argsObj["pm"]
    print(f"interp_steps1 is {interp_steps1}, and usePM (use ParamManager) is {usePM}")

    for p in range(minpitch, maxpitch+1) :

        # linear
        #gen_batch, latents = eval_manager.test_single_pitch_latent_interpolation(p_val=p, z0=z0, z1=z1, steps=10)
        #sperical
        #gen_batch, latents = eval_manager.qslerp(pitch=p, z0=z0, z1=z1, steps=10)
        #staggred
        gen_batch, latents = eval_manager.test_single_pitch_latent_staggered_interpolation(p_val=p, z0=z0, z1=z1, steps=interp_steps1, d1nvar=argsObj["d1nvar"], d1var=argsObj["d1var"])


        audio_out = map(postprocess, gen_batch)

        if not usePM :  #then just output as usual, including option to write latents if provided
            saveAudioBatch(audio_out,
                           path=output_dir,
                           basename='test_pitch_sweep'+ "_"+str(p), 
                           sr=config["transformConfig"]["sample_rate"],
                           latents=latents)

        else:                       # save paramManager files, (and don't write latents separately)
            data=list(audio_out) #LW it was a map
            zdata=zip(data,latents) #zip so we can enumerate through pairs of data/latents

            istep=0
            vstep=0

            for i, (audio, params) in enumerate(zdata) :

                istep=int(i/argsObj["d1nvar"])
                vstep=(vstep+1)%argsObj["d1nvar"]

                if type(audio) != np.ndarray:
                    audio = np.array(audio, float)

                path=output_dir
                basename='test_pitch_sweep'+ "_"+str(p) 
                sr=config["transformConfig"]["sample_rate"]

                #foo=f'{basename}_{istep}_{vstep}.wav'

                out_path = os.path.join(path, f'{basename}_{istep}_{vstep}.wav')
                # paramManager, create 
                pm=paramManager.paramManager(out_path, output_dir)  ##-----------   paramManager  interface ------------------##
                #param_out_path = os.path.join(path, f'{basename}_{i}.params')
                pm.initParamFiles(overwrite=True)


                if not os.path.exists(out_path):
                    #write_wav(out_path, audio.astype(float), sr)
                    sf.write(out_path, audio.astype(float), sr)

                    duration=len(audio.astype(float))/float(sr)
                    #print(f"duration is {duration}")
                    if latents != None :
                        pm.addParam(out_path, "pitch", [0.0,duration], [(p-minpitch)/pitchrange,(p-minpitch)/pitchrange], units="norm, midip in[58,70]", nvals=0, minval='null', maxval='null')
                        pm.addParam(out_path, "instID", [0.0,duration], [istep/interp_steps1norm,istep/interp_steps1norm], units="norm, interp steps in[0,10]", nvals=10, minval='null', maxval='null')
                        #pm.addParam(out_path, "envPt", [0.0,duration], [0,1.0], units=f"norm, duration in[0,{duration}]", nvals=0, minval='null', maxval='null')
                    
                        segments=11 # to include a full segment for each value including endpoints
                        envTimes, envVals=makesteps(np.linspace(0,duration,segments+1,True) , np.linspace(0,1,segments,True)) #need one extra time to flank each value
                        pm.addParam(out_path, "envPt", envTimes, envVals, units=f"norm, duration in[0,{duration}]", nvals=0, minval='null', maxval='null')

                        # write paramfile 
                        #torch.save(params, param_out_path)
                        #np.savetxt(txt_param_out_path, params.cpu().numpy())
                else:
                    print(f"saveAudioBatch: File {out_path} exists. Skipping...")
                    continue





    print("FINISHED!\n")