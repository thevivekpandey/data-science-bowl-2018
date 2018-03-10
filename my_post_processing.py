import sys
from handle_overlap import handle_overlap
from remove_extraneous_masks import remove_extraneous_masks

if __name__ == '__main__':
    in_file_name = sys.argv[1]
    assert '.1.csv' in in_file_name
    base_name = in_file_name[:-6]

    step1_output = base_name + '.2.csv'
    remove_extraneous_masks(in_file_name, step1_output)

    step2_output = base_name + '.3.csv'
    handle_overlap(step1_output, step2_output)
    
