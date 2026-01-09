# benthic-habitat-segmentation-ML
Date Assigned: October 29, 2025
Date Submitted: December 9, 2025

This project was an assignment for CISC484: Intro to Machine Learning, which is being taught in the Fall of 2025 at the University of Delaware by Professor Xu Yuan. This class is an undergraduate course within the Computer and Information Sciences department at the University of Delaware College of Engineering. The objective of this course is to introduce students to the key concepts and techniques of machine learning.

The objective of this project was for students to develop a deep learning model for benthic habitat segmentation and classification, through WorldView-3 imagery at Guam. Agana Bay and Manell-Geus were the two regions of Guam that were studied for this project.

There were three main challenges for this deep learning project. First, the dataset exhibited severe class imbalance between the 7 habitats, as the Land and Ocean habitats accounted for 65.3% of all pixels. Second, the images were of variable sizes, with some being as small as 528x447px and others being as large as 1370x2051px. Finally, the images came in 8 different spectral bands. 

Some demo code was provided to students by Professor Yuan. This code used a VGG model for a deep learning task with MNIST images. For the benthic habitat segmentation task, I created two different models. First, I attempted to modify the VGG model from the demo code in order to fit this benthic habitat segmentation and classification task. Second, I developed a U-NET model using an online YouTube video, which is mentioned in the report. The U-NET model performed better in training than the VGG model, so I used just the U-NET model for the predictions step.

The 'Project 4 Code' folder shows my Jupyter Notebook and Python work for this project. Libraries/packages of interest include PyTorch, Tensorflow, MatPlotLib, NumPy, and more. The training results for both models were given their own folder within this repository. For the U-NET model, folders exist for the prediction images and prediction results. A few of the .pth files for this assignment were too large to upload through GitHub, which is something I am currently looking to account for. This was particularly a problem for the U-NET model. Finally, I encourage a look at the 'Project 4 Report' folder, which contains the report I submitted for this assignment. Within this report, my methodology, reasoning, and results are discussed.

Professor Yuan's University of Delaware page: https://www.cis.udel.edu/people/faculty/xu-yuan/