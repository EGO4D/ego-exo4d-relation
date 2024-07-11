SCRIPT=scripts/preprocess_corres.py
TAKE_IDS=/media/ys/storage/data/egoexo/project/3dlg-hcvc/egoexo4d/meta_relations_data/valid_takes.txt
PARALLEL_N=2

echo "===================================="
echo "Exporting point clouds ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $SCRIPT \
    --take {1} " :::: $TAKE_IDS