source /DATA/Aurindum/conda/etc/profile.d/conda.sh; conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /DATA/Aurindum/GPUSwarm/VectorFitting
L=/tmp/vf_sweep.log; : > $L
echo "### decap_classify.py $(date)" >> $L; ( time python decap_classify.py ) >> $L 2>&1
for p in ddr21 ddr_full; do echo "### pdn_poles VDDQ_DDR3 $p order 24" >> $L; python pdn_poles.py --board VDDQ_DDR3 --prob $p --order 24 --hi 1e8 --plot >> $L 2>&1; done
for p in mer1 mer2 pll1v0 pll1v8 mphyvdd mphy_full; do echo "### pdn_poles MPHY $p order 24" >> $L; python pdn_poles.py --board MPHY --prob $p --order 24 --hi 1e8 --plot >> $L 2>&1; done
echo "### done $(date)" >> $L
