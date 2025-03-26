ls output/CEU_neutral*.trees > all_CEU_neutral_trees.txt

for i in 001 002 003 004; do
    ls output/CEU_sel_${i}*.trees > all_CEU_sel_${i}_trees.txt
done