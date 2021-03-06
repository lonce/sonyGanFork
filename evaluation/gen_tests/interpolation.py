import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioPreprocessor
import pdb

def generate(parser):
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # check if conditioning attribute is present
    if 'attribute_list' in config['loaderConfig'].keys():
        condition_parameter = config['loaderConfig']['attribute_list'][0]
    else:
        print("There is no conditioning parameter ('attribute_list' is empty). Exiting!")
        exit(0)
    # We load a dummy data loader for post-processing
    postprocess = AudioPreprocessor(**config['transformConfig']).get_postprocessor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "interpolation")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=100)
    gen_batch, latents = eval_manager.test_single_pitch_latent_interpolation(condition_parameter)  #this function does take args ....
    output_path = mkdir_in_path(output_dir, f"one_z_pitch_sweep")
    audio_out = map(postprocess, gen_batch)
    
    saveAudioBatch(audio_out,
                   path=output_path,
                   basename='test_pitch_sweep', 
                   sr=config["transformConfig"]["sample_rate"],
                   latents=latents)
    print("FINISHED!\n")