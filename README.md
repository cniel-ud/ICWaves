Welcome to the ICWaves project repo! (this readme written by Austin. Ask me if you have any questions)

To get an overview of the theoretical aspects of the project, rationale & justification:

Start by reading the relevant chapters of Carlos' Mendoza's dissertation.
The base approach taken so far is detailed in Ch. 5 - Bag-of-waves for IC Labeling.
One of the ablations that we will start with first is to switch out his shift-invariant k-means algorithms
(detailed in the paper here: http://arxiv.org/abs/2108.03177) with his quick shift matrix profile (QSMP)
algorithm. The QSMP work is detailed in Ch. 4 - Density-guided waveform learning.

Other resources to get a grip on the project so far would be to look at the non-archival ICLR workshop
paper we wrote and the poster for it. I'll send this stuff to y'all (not up on the arxiv) when you get going
on the slack channel.


-----------------------------------------------------------------------
To get started on the code:

The way Carlos has set this up is with scientific mode in Pycharm enabled. This is only in Pycharm Professional
edition, but the good news is that you get a free license for it with your udel.edu email address. So get that
set up first. Once you're in Pycharm, you can go in and out of scientific view by clicking 'view' above.

Additionally, you'll need to be added to the lab Github group. Let me know and I'll take care of it.

There are several directories you should make in your project, just don't add them to github. These will store
the images data (img and data are their names, respectively). 

Next you'll need to get the EEG data onto your computer. Use the shell script in the data folder provided in this branch.
It's called 'ds003004-1.1.1.sh'. Run this in the Pycharm terminal with 'bash ds003004-1.1.1.sh' when you're
in the ICWaves/data directory.
Note that this will download the entire dataset into your directory, which is about 40 GB. If you want individual
files, you can use datalad through the command line or write a python script with the datalad api to do so.
I'm intending to write the rest of the project code assuming that all data is there though.

Run the matlab files on the data. Then you'll have data that's been processed with IC labels and whatnot.
From this point try the example Jupyter notebooks. 