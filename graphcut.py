import argparse
from cutui import CutUI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Interactive Graph Cut",
                                     description="Interactively segment an image", add_help=True)
    
    parser.add_argument('-i', '--infile', help='Input image file to segment.', required=True)
    parser.add_argument('-o', '--outfile', help='Used to save segmented images.', required=True)

    args = parser.parse_args()

    ui = CutUI(args.infile, args.outfile)
    ui.run()