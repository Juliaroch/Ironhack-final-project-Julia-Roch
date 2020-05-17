#main.py
import argparse
#from plots import plots
from demo import demo

def main(path_name, n):
    #x, y = plots(n)
    demo(path_name, n)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path_to_images', default=None, required=True, type=str, help="""Please provide the path for your paintings:""")
    parser.add_argument('-n', '--number', dest='number_predictions', default=None, required=True, choices=range(0, 100), type=int, help="""Please provide the number of predictions you would like to make:""")
    args = parser.parse_args()
    path_name = args.path_to_images
    n = args.number_predictions
    print('''ART-ificial intelligence is working on mixing science and art, it won't take very long...''')

    main(path_name, n)
