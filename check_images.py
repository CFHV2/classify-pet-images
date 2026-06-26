#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Carlos Fernandez De La Hoz Valiente
# DATE CREATED: 18/06/2026                                 
# REVISED DATE: 19/06/2026 
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images_solution.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():

    # Init start time
    start_time = time()

    # 1. Arguments
    in_arg = get_input_args()
    print("DIR:", in_arg.dir)
    print("MODEL:", in_arg.arch)
    print("DOGFILE:", in_arg.dogfile)

    # 2. Get Real Labels
    results_dic = get_pet_labels(in_arg.dir)
    print("Number of items in dictionary: ", len(results_dic))

    # 3. Call classify_images
    classify_images(in_arg.dir, results_dic, in_arg.arch)

    # 4. Separate Dogs from no dogs
    adjust_results4_isadog(results_dic, in_arg.dogfile)
    for key in results_dic:
        print(results_dic[key])
        break

    # 5. Calculate Statistics
    results_stats = calculates_results_stats(results_dic)

    # 6. Print the results
    print_results(results_dic, results_stats, in_arg.arch, True, True)

    # End Time and Total Time
    end_time = time()
    tot_time = end_time - start_time

    # Format hh:mm:ss
    hours = int(tot_time / 3600)
    min = int((tot_time % 3600) / 60)
    #segundos = int((tot_time % 3600) % 60)
    seg = round((tot_time % 3600) % 60)

    print("\nTotal Time Elapsed: {}:{}:{}".format(hours, min, seg))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()