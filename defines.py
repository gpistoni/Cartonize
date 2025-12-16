import os

############################################################################################################################################################
datasetName = "Cartonize"
block_size = 32
batch_size = 2000

# Model deep
n_layer1=2
n_layer2=2

checkpoint_resume = False
lr = 2e-4

############################################################################################################################################################
root_dir =  os.path.join('/' ,'home', 'giulipis' ,'Dataset' , datasetName)

############################################################################################################################################################
fullImage_dir = os.path.join(root_dir, "Fullimages")

sample_dir = os.path.join(root_dir, "Train_Samples")
sample_dir_A =  os.path.join(sample_dir, "A") 
sample_dir_B =  os.path.join(sample_dir, "B") 

trainOutput_dir = os.path.join(root_dir, "Train_Output" )

models_dir = os.path.join(root_dir, "Models")

checkpoint_file = os.path.join(models_dir, "checkpoint_" + str(n_layer1) + "_" + str(n_layer2) + ".pth")