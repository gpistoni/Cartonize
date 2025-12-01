import os

############################################################################################################################################################
datasetName = "Cartonize"
block_size = 64
batch_size = 1 + int(1e6 / (block_size*block_size))

############################################################################################################################################################
root_dir =  os.path.join('/' ,'home','giulipis' ,'Dataset' , datasetName)

############################################################################################################################################################
fullImage_dir = os.path.join(root_dir, "Fullimages")

sample_dir = os.path.join(root_dir, "Train_Samples")
sample_dir_A =  os.path.join(sample_dir, "A") 
sample_dir_B =  os.path.join(sample_dir, "B") 

trainOutput_dir = os.path.join(root_dir, "Train_Output" )

checkpoint_file ="checkpoint.pth"
