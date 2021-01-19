import importlib
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generation testing script', add_help=False)
    parser.add_argument('generation_test', type=str,
                        help='Name of the generation test to launch. To get \
                        the arguments specific to an generation test please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('-d', '--dir', help="Path to model's root folder",
                        type=str, dest="dir")
    parser.add_argument('-n', '--nsynth-path', help="Path to nsynth dataset root folder",
                        type=str, dest="nsynth_path")
    parser.add_argument('-o', '--out-dir', help='Output directory',
                        type=str, dest="outdir", default="output_networks")
    parser.add_argument('-m', '--midi', help='Path to midi file',
                        type=str, dest="midi", default="./test_midi_files/midi_furelisa_adapted.mid")
    
    #These are all for 2D generation 
    parser.add_argument('--p0', help='param0 for endpoint of dimension 1 (for 2D)',
                        type=int, dest="p0", default=55)
    parser.add_argument('--p1', help='param1 for endpoint of dimension 1 (for 2D)',
                        type=int, dest="p1", default=55)
    parser.add_argument('--z0', help='z0 filename (for 2D)',
                        type=str, dest="z0", default="./output/z0.txt")
    parser.add_argument('--z1', help='z1 filename (for 2D)',
                        type=str, dest="z1", default="./output/z1.txt")
    parser.add_argument('--d1', help='discretization along first dimension, p0...p1, including endpoint (for 2D)' ,
                        type=int, dest="d1", default=13)
    parser.add_argument('--d2', help='discretization along second dimension z0...z1, including endpoint (for 2D)',
                        type=int, dest="d2", default=10)




    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    module = importlib.import_module("evaluation.gen_tests." + args.generation_test)
    print("Running " + args.generation_test)

    parser.add_argument('-h', '--help', action='help')
    out = module.generate(parser)

    if out is not None and not out:
        print("...FAIL")

    else:
        print("...OK")
