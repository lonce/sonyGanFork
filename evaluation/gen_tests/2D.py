import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioPreprocessor


def generate(parser):
    args = parser.parse_args()

    argsObj=vars(args)
    print(f"generate args: {argsObj}")

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    postprocess = AudioPreprocessor(**config['transformConfig']).get_postprocessor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "2D")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=2)

    z0=torch.load(argsObj["z0"])
    z1=torch.load(argsObj["z1"])

    for p in range(int(argsObj["p0"]), int(argsObj["p1"])+1) :
        print(f"generate interps for {p}")
        # linear
        #gen_batch, latents = eval_manager.test_single_pitch_latent_interpolation(pitch=p, z0=z0, z1=z1, steps=10)
        gen_batch, latents = eval_manager.qslerp(pitch=p, z0=z0, z1=z1, steps=10)
        # sperical
        audio_out = map(postprocess, gen_batch)

        saveAudioBatch(audio_out,
                       path=output_dir,
                       basename='test_pitch_sweep'+ "_"+str(p), 
                       sr=config["transformConfig"]["sample_rate"],
                       latents=latents)



    print("FINISHED!\n")