# Download the mini-vg dataset

mkdir -p data/

echo 'Downloading the mini-vg dataset...'
wget http://svl.stanford.edu/projects/scene-graph/dataset/mini-vg.zip
unzip mini-vg.zip
mv vg data/
rm mini-vg.zip

echo 'Downloading the final model checkpoint...'
wget http://svl.stanford.edu/projects/scene-graph/checkpoints/dual_graph_vrd_final_iter2_checkpoint.zip
unzip dual_graph_vrd_final_iter2_checkpoint.zip
mv dual_graph_vrd_final_iter2 checkpoints
rm dual_graph_vrd_final_iter2_checkpoint.zip

echo 'done'
