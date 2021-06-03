# rsync outputs from cbica
rsync -avzh parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/pfactor_gradients/output_cluster \
/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/

# rsync data to cbica
rsync -avzh /Volumes/T7/research_data/pnc/processedData \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_data/pnc/

rsync -avzh /Volumes/T7/research_data/parcellations \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_data/

rsync -avzh /Volumes/T7/research_data/fukutomi_2018ni_brain_maps \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_data/
