import sys
from handle_overlap import handle_overlap
from remove_extraneous_masks import remove_extraneous_masks
from handle_siblings import handle_siblings

if __name__ == '__main__':
    #There are 6 file names derived from base file name, for 6 mutations
    base_file_name = sys.argv[1]

    step1_output = base_file_name + '.ver9.1.csv'
    handle_siblings(base_file_name, step1_output)
    print('Done with step 1')

    step2_output = base_file_name + '.ver9.2.csv'
    remove_extraneous_masks(step1_output, step2_output)
    print('Done with step 2')

    step3_output = base_file_name + '.ver9.3.csv'
    handle_overlap(step2_output, step3_output)
    print('Done with step 3')
