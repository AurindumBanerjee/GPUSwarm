source /DATA/Aurindum/conda/etc/profile.d/conda.sh; conda activate work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /DATA/Aurindum/GPUSwarm/VectorFitting
mkdir -p out_order24 out_order20 out_order32
cp poles_*.json poles_*.png decap_classes.json out_order24/
L=/tmp/vf_extra.log; : > $L
echo "### ddr21 order 20 (reference setting)" >> $L; python pdn_poles.py --board VDDQ_DDR3 --prob ddr21 --order 20 --hi 1e8 --plot >> $L 2>&1; cp poles_VDDQ_DDR3_ddr21.* out_order20/
for spec in "VDDQ_DDR3 ddr21" "MPHY mer1" "MPHY mer2" "MPHY pll1v0" "MPHY mphyvdd"; do set -- $spec
  echo "### $1 $2 order 32" >> $L; python pdn_poles.py --board $1 --prob $2 --order 32 --hi 1e8 --plot >> $L 2>&1; cp poles_$1_$2.* out_order32/; done
cp out_order24/poles_*.json out_order24/poles_*.png .     # canonical (order-24) outputs back in place
python - >> $L 2>&1 <<PY
import numpy as np, scipy.io as sio, json
D = "/DATA/Aurindum/GPUSwarm/Data/"
print("### sampled |Z11| peaks picked from the data (110 Hz - 100 MHz): top local maxima by value, and nearest pole at order 20/24/32")
def peaks(board, prob):
    y = np.transpose(sio.loadmat(D + f"{board}/y_sp_{prob}.mat")["y"], (2, 0, 1)); f = sio.loadmat(D + f"{board}/freq_sp.mat")["freq"].ravel()
    z = np.abs(np.linalg.inv(y)[:, 0, 0]); m = (f <= 1e8)
    f, z = f[m], z[m]; i = np.where((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
    i = i[np.argsort(z[i])[::-1]][:6]; return f[i], z[i], f
def poles(path):
    j = json.load(open(path)); k = [p for p in j.keys()]; return j
for board, prob in (("VDDQ_DDR3", "ddr21"), ("MPHY", "mer1"), ("MPHY", "mer2")):
    fp, zp, fgrid = peaks(board, prob)
    print(f"{board}/{prob}: grid spacing near 25 MHz = {np.diff(fgrid[(fgrid>2e7)&(fgrid<3e7)]).mean()/1e3:.1f} kHz")
    for a, b in zip(fp, zp): print(f"   sampled peak {a:.5g} Hz  |Z| {b:.4g} ohm")
j = json.load(open("out_order24/poles_VDDQ_DDR3_ddr21.json")); print("json keys:", list(j.keys())[:12])
PY
echo "### done" >> $L
