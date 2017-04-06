# Download the mini-vg dataset

mkdir -p data/

echo 'Downloading the mini-vg dataset...'
wget http://cvgl.stanford.edu/scene-graph/dataset/mini-vg.zip
unzip mini-vg.zip
mv vg data/
rm mini-vg.zip

echo 'Downloading the final model checkpoint...'
wget http://cvgl.stanford.edu/scene-graph/checkpoints/dual_graph_vrd_final_iter2_checkpoint.zip
unzip dual_graph_vrd_final_iter2_checkpoint.zip
mv dual_graph_vrd_final_iter2 checkpoints
rm dual_graph_vrd_final_iter2_checkpoint.zip

echo 'done'
