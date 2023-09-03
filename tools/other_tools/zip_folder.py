import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='zip files since this was removed for some reason')
    parser.add_argument('--output_filename', help='path of endovis_2017_processed_folder')
    parser.add_argument('--dir_name', help='path of the temporary directory')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    output_filename = args.output_filename 
    dir_name = args.dir_name
    shutil.make_archive(output_filename, 'zip', dir_name)


if __name__ == '__main__':
    main()