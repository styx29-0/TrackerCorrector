import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'nfs' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot'
subject_name = 'fanyanlong' # "none" "fanyanlong" "liweilong" "lupengyi" "yinxiaozhe"

# seqtrack, stark_st, hiptrack, siamfc
# seqtrack_b256, stark_st2, hiptrack, siamfc
trackers.extend(trackerlist(name='siamfc', parameter_name='siamfc', dataset_name=dataset_name,
                            run_ids=5, display_name='siamfc'))

dataset = get_dataset(dataset_name, subject_name=subject_name)

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

