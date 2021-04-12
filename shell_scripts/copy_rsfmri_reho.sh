server_dir=/data/joy/BBL/studies/pnc/processedData/restbold/restbold_201607151621/
out_dir=/data/jag/bassett-lab/lindenmp/pnc/processedData/restbold/restbold_201607151621/

# Schaefer
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}reho/roi/SchaeferPNC/; cp ${server_dir}${i}${j}reho/roi/SchaeferPNC/*reho*.1D ${out_dir}${i}${j}reho/roi/SchaeferPNC/; done; done
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}reho/roi/Schaefer200PNC/; cp ${server_dir}${i}${j}reho/roi/Schaefer200PNC/*reho*.1D ${out_dir}${i}${j}reho/roi/Schaefer200PNC/; done; done

# Lausanne
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}reho/roi/Lausanne125/; cp ${server_dir}${i}${j}reho/roi/Lausanne125/*reho*.1D ${out_dir}${i}${j}reho/roi/Lausanne125/; done; done
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}reho/roi/Lausanne250/; cp ${server_dir}${i}${j}reho/roi/Lausanne250/*reho*.1D ${out_dir}${i}${j}reho/roi/Lausanne250/; done; done

# Glasser
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}reho/roi/GlasserPNC/; cp ${server_dir}${i}${j}reho/roi/GlasserPNC/*reho*.1D ${out_dir}${i}${j}reho/roi/GlasserPNC/; done; done
