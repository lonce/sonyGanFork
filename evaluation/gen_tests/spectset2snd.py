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
    output_dir = mkdir_in_path(output_dir, "2D_spectset")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))


    gen_batch, latents=torch.load(argsObj["gen_batch"])


    interp_steps0=int(argsObj["d0"])
    interp_steps0norm=interp_steps0 -1 # because the batch generater will spread the steps out to include both endpoints

    interp_steps1=int(argsObj["d1"])
    interp_steps1norm=interp_steps1 -1 # because the batch generater will spread the steps out to include both endpoints


    usePM=argsObj["pm"]
 
    g=list(gen_batch)

    assert interp_steps0*interp_steps1==len(g), f"product of d0, d1 interpolation steps({interp_steps0},{interp_steps1}) != batch length ({len(g)})"



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

        #d1nvar=argsObj["d1nvar"]
        d1nvar=1 # no variations for this spectset generation

        rowlength=interp_steps0*d1nvar
        print(f'rowlength is {rowlength}')

        for k, (audio, params) in enumerate(zdata) :
            istep = int(k/rowlength)  #the outer counter, orthogonal to the two lines defining the submanifold

            j=k%rowlength
            jstep=int(j/d1nvar)
            vstep=(vstep+1)%d1nvar

            print(f'doing row {istep}, col {jstep}, and variation {vstep}')

            if type(audio) != np.ndarray:
                audio = np.array(audio, float)

            path=output_dir
            basename='test_spectset2snd' 
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