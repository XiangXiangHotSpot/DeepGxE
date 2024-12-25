#1
gunzip F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates.vcf.imputed.CIMMYT.vcf.gz
wait

#2
cat F_MAF0.01_Miss50_Het10-Merged.all.discover.lines.and.selection.candidates.vcf.imputed.CIMMYT.vcf \
| grep -v "^##" \
| cut -f3,10- \
| sed 's/0\/0/0/g' \
| sed 's/1\/1/2/g' \
| sed 's/0\/1/1/g' \
| sed 's/1\/0/1/g' \
| sed 's/\.\/\./-1/g' \
| tr "\t" "," > ../output/genotype.csv

#3
head -n 1 genotype.csv | sed 's/,/ /g' > ../output/genotype_ID.txt

