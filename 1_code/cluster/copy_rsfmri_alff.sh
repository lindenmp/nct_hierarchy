server_dir=/data/joy/BBL/studies/pnc/processedData/restbold/restbold_201607151621/
out_dir=/data/jag/bassett-lab/lindenmp/pnc/processedData/restbold/restbold_201607151621/

# Schaefer
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}alff/roi/SchaeferPNC/; cp ${server_dir}${i}${j}alff/roi/SchaeferPNC/*alff*.1D ${out_dir}${i}${j}alff/roi/SchaeferPNC/; done; done
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}alff/roi/Schaefer200PNC/; cp ${server_dir}${i}${j}alff/roi/Schaefer200PNC/*alff*.1D ${out_dir}${i}${j}alff/roi/Schaefer200PNC/; done; done

# Lausanne
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}alff/roi/Lausanne125/; cp ${server_dir}${i}${j}alff/roi/Lausanne125/*alff*.1D ${out_dir}${i}${j}alff/roi/Lausanne125/; done; done
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}alff/roi/Lausanne250/; cp ${server_dir}${i}${j}alff/roi/Lausanne250/*alff*.1D ${out_dir}${i}${j}alff/roi/Lausanne250/; done; done

# Glasser
cd ${server_dir}
for i in */; do cd ${server_dir}${i}; for j in */; do echo ${i}${j}; mkdir -p ${out_dir}${i}${j}alff/roi/GlasserPNC/; cp ${server_dir}${i}${j}alff/roi/GlasserPNC/*alff*.1D ${out_dir}${i}${j}alff/roi/GlasserPNC/; done; done