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
    if argsObj["z2"] == None :
        z2=0
    else :
        z2=torch.load(argsObj["z2"])
    if argsObj["z3"] == None : 
        z3 = 0
    else :
        z3=torch.load(argsObj["z3"])


    interp_steps0=int(argsObj["d0"])
    interp_steps0norm=interp_steps0 -1 # because the batch generater will spread the steps out to include both endpoints

    interp_steps1=int(argsObj["d1"])
    interp_steps1norm=interp_steps1 -1 # because the batch generater will spread the steps out to include both endpoints

    usePM=argsObj["pm"]
    print(f"interp_steps0 is {interp_steps0}, interp_steps1 is {interp_steps1}, and usePM (use ParamManager) is {usePM}")



    #######   ---- unconditioned 
    gen_batch, latents = eval_manager.unconditioned_linear_interpolation(line0z0=z0, line0z1=z1, line1z0=z2, line1z1=z3, d0steps=interp_steps0, d1steps=interp_steps1, d1nvar=argsObj["d1nvar"], d1var=argsObj["d1var"])


    audio_out = map(postprocess, gen_batch)

    if not usePM :  #then just output as usual, including option to write latents if provided
        saveAudioBatch(audio_out,
                       path=output_dir,
                       basename='test_2D4pt', 
                       sr=config["transformConfig"]["sample_rate"],
                       latents=latents)

    else:                       # save paramManager files, (and don't write latents separately)
        data=list(audio_out) #LW it was a map, make it a list
        zdata=zip(data,latents) #zip so we can enumerate through pairs of data/latents


        vstep=-1  # gets incremented in loop

        rowlength=interp_steps0*argsObj["d1nvar"]
        print(f'rowlength is {rowlength}')

        for k, (audio, params) in enumerate(zdata) :
            istep = int(k/rowlength)  #the outer counter, orthogonal to the two lines defining the submanifold

            j=k%rowlength
            jstep=int(j/argsObj["d1nvar"])
            vstep=(vstep+1)%argsObj["d1nvar"]

            print(f'doing row {istep}, col {jstep}, and variation {vstep}')

            if type(audio) != np.ndarray:
                audio = np.array(audio, float)

            path=output_dir
            basename='test_2D4pt' 
            sr=config["transformConfig"]["sample_rate"]

            #foo=f'{basename}_{jstep}_{vstep}.wav'

            out_path = os.path.join(path, f'{basename}_d1.{istep}_d0.{jstep}_v.{vstep}.wav')
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
                    #pm.addParam(out_path, "dim1", [0.0,duration], [(p-minpitch)/pitchrange,(p-minpitch)/pitchrange], units="norm, midip in[58,70]", nvals=0, minval='null', maxval='null')
                    pm.addParam(out_path, "dim0", [0.0,duration], [jstep/interp_steps0norm,jstep/interp_steps0norm], units=f'norm, interp steps in[0,{interp_steps0}]', nvals=interp_steps0, minval='null', maxval='null')
                    if interp_steps1norm > 0 : #else just doing 1D interpolation
                        pm.addParam(out_path, "dim1", [0.0,duration], [istep/interp_steps1norm,istep/interp_steps1norm], units=f'norm, interp steps in[0,{interp_steps1}]', nvals=interp_steps1, minval='null', maxval='null')
                
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