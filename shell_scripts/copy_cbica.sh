# rsync outputs from cbica
rsync -avzh parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/pfactor_gradients/output_cluster \
/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/

# rsync data to cbica
rsync -avzh /Volumes/T7/research_data/PNC/processedData \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_data/PNC/

# scp -r parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/pfactor_gradients/output_cluster \
# /Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/