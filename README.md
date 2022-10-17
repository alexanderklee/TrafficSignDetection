# Detecting Irregularites in Large Image Datasets for Machine Learning

## Overview

Producing high quality machine learning (ML) test datasets is paramount to insure accurate machine learning models. The process for producing, annotating/labeling, and audting images is manageable when the image dataset is small. However, if the image datasets are in the 100K+ range then this process gets unwieldly and prone to error. The net results are the machine learning models will not be trained properly due to errors in the dataset. To overcome the dataset scale, the approach is to take a programmatic approach to auditing annotattions/labels and the images to help assist the auditor in the data labeling process.

## Goal

To address the data labeling process at scale, software is required to help automate the tasks typically performed by humans in the loop (HITL). This program leverages Python and the Scale.ai API's to help automate the auditing process. This script uses the `ScaleClient` to addess the Scale.ai infrastructure to extract project, task, and dataset information that will help facilitate the auditing tasks programmatically. Other libraries like `Shapely` and `cv2` are used to perform bounding box (bbox) operations and blur detection. Lastly, this script uses basic geometry and inference to detect (or warn) if annotations/labels are suspicious enough to be flagged.

## Code Overview

The code is split across seven (7) files. These files and their descriptions are as follows:

- `quality-test.py`: This is the `main` function that accesses the Scale.ai projects & tasks and calls all other functions for testing
- `img.py`: This function parses a URL & extracts the image name, fetches the image, stores it on disk and returns information about the image
- `iou.py`: This function calculates the intersection-over-union (IOU) between two bounding boxes
- `flags.py`: This function processes all the test flags, produces a log dictionary, and writes the results to disk
- `menu.py`: This function produces a terminal menu to allow the user to select which Scale.ai project to process
- `helper.py`: This file contains the many helper functions used by other functions in this program
- `torch-iou.py`: (deprecated) Implemented IOU manually using torch and tensors but the `Shapely` module made this easier
- `handler`: This is a function declared inside `main` to catch SIGINT (ctrl-c) and exit the program safely. `handler` needs context from `main` and therefore cannot be externally defined.

## Run the Code

To run this program, you will need a `Live` Scale.ai API key and is a required command-line (CLI) argument to this program.

`# python3 quality-test.py --key "live_XXXXXXXXXXXXXXXXXXXXXXXXX"`

Assuming the key is valid, you will see a list of menu items. These are project names that the key is associated with. This is what the menu looks like:

```
------------------------------------------------------------------------------------
-                                Scale Project Menu                                -
------------------------------------------------------------------------------------
[0]: sunshineTest                               [1]: sunshineTestTwo
[2]: Traffic Sign Detection                     [3]: Sample Accounts Payable Invoices
[4]: Test Project                               [5]: Test_Project5
.....
------------------------------------------------------------------------------------
Enter your option (whole numbers only): 2
```

Once you enter the number associated with the project your interested in, the program will process all tasks and perform tests against each task & image. A progress bar will appear to allow the auditor to track progress:

`Processing Tasks: 62%|███████████████████████████████▎ | 5/8 [00:03<00:01, 1.71it/s]`

Once completed, a message will appear letting the auditor know where the output files are located:

`Completed!`
`Images downloaded to the images/ directory.`
`Log files written to the responses/ directory.`

## Output Files

The script creates three directories. The first directory is a parent directory named after the **project name**. The other two directories are sub-directories where one is called `images` and the other is `responses`. The `images` directory is where the original images are stored and can be used by auditors to quickly inspect images. The `responses` directory is a list of files, named using the `task_id` and contains a brief overview of the test status in JSON format.

## Improvements

Where this project can improve is including more test scenarios. The scenarios I would like to add are detectors for bright/dark spots, flashes, reflections, camoflauged objects, and using a better blur detection scheme like Fast-Fourier Transform (FFT). Beyond adding more quality tests, the program itself could be better constructed using more object-oriented approaches.
