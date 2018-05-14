import sys
import os


def main(folder_name):
    target_folder = '/archive/wkessels/output/{}/'.format(folder_name)
    os.system('cd ~')
    os.system('python analyse_out_semantics.py {}'.format(folder_name))
    os.system('python analyse_out_features.py {}'.format(folder_name))
    os.system('python dim_reduction.py {}'.format(folder_name))
    os.system('python plot_ROC.py {}'.format(folder_name))

if __name__ == '__main__':
    if len(sys.argv) == 4:
        [main(sys.argv[i]) for i in range(1, len(sys.argv))]
        os.system('python combine_ROCs.py {} {} {}'.format(sys.argv[1], sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 2:
        [main(sys.argv[i]) for i in range(1, len(sys.argv))]
    elif len(sys.argv) == 1:
        raise IOError('No input argument given.')
