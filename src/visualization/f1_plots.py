import matplotlib.pyplot as plt
import os
import logging
import argparse

parser = argparse.ArgumentParser(description='plotting the b-cube f1 score')

parser.add_argument('--m_type', type=str,
                    help='For which type plot the score - event\entity')

args = parser.parse_args()

models_dir = 'models'

results_file = 'epochs_scores.txt'

def plot_dirs_f1(models_dir, plt_path):
    dirs = os.listdir(models_dir)
    for d in dirs:
        if results_file not in os.listdir(os.path.join(models_dir, d)):
            continue
        f1_path = os.path.join(models_dir, d, results_file)
        event, entity = extract_f1s(f1_path)
        scores = event if args.m_type.lower() == 'event' else entity
        plt.plot(range(1, len(scores)+1), scores, '-', label='{}, max:{}'.format(d, max(scores))) 
    plt.legend()
    plt.xlabel('epoc #')
    plt.ylabel('f1 bcub score')
    plt.title('{} f1 score'.format(args.m_type.lower()))
    plt.savefig(plt_path)
    
def extract_f1s(path):
    event = []
    entity = []
    with open(path, 'r') as f:
        for l in f:
            d = l.strip().split()
            if len(d) == 0:
                break
            event.append(float(d[5]))
            entity.append(float(d[13]))   
    return event, entity
    
if __name__ == '__main__':
    plot_dirs_f1(models_dir, '{}.png'.format(args.m_type.lower()))