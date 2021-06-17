import importlib
import argparse
import sys

if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Generation testing script', add_help=False)
    #parser.add_argument('generation_test', type=str,
    #                    help='Name of the generation test to launch. To get \
    #                    the arguments specific to an generation test please \
    #                    use: eval.py evaluation_name -h')
    parser.add_argument('-d', '--dir', help="Path to model's root folder",
                        type=str, dest="dir")
    parser.add_argument('-n', '--nsynth-path', help="Path to nsynth dataset root folder",
                        type=str, dest="nsynth_path")
    parser.add_argument('-o', '--out-dir', help='Output directory',
                        type=str, dest="outdir", default="output_networks")

    parser.add_argument('--d0', help='discretization along first dimension, p0...p1, including endpoint (for 2D)' ,
                    type=int, dest="d0", default=1)

    parser.add_argument('--d1', help='discretization along second dimension z0...z1, including endpoint (for 2D)',
                        type=int, dest="d1", default=1)


    #These are all for 2D generation 
    parser.add_argument('--gen_batch', help='file name of batch of spectra as produced from GAN and latents (batch, latents)',
                        type=str, dest="gen_batch", default="gen_batch_file")
   

    parser.add_argument("--pm", type=str2bool, nargs='?', dest="pm", const=True, default=False,
                        help="create paramManager files.")




    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    #module = importlib.import_module("evaluation.gen_tests." + args.generation_test)
    module = importlib.import_module("evaluation.gen_tests." + "spectset2snd")
    #print("Running " + args.generation_test)
    print("Running ")

    parser.add_argument('-h', '--help', action='help')
    out = module.generate(parser)

    if out is not None and not out:
        print("...FAIL")

    else:
        print("...OK")
