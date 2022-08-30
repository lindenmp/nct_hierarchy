# rsync code to cbica
rsync -avzh /Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/src \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/nct_hierarchy/
rsync -avzh /Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/scripts \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/nct_hierarchy/

# rsync data to cbica
rsync -avzh /Users/lindenmp/research_data \
parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/

# rsync outputs from cbica
mkdir -p /Users/lindenmp/research_projects/nct_hierarchy/output_cluster
rsync -avzh --exclude '*gmat*' parkesl@cubic-login.uphs.upenn.edu:/cbica/home/parkesl/research_projects/nct_hierarchy/output_cluster/ \
/Users/lindenmp/research_projects/nct_hierarchy/output_cluster
