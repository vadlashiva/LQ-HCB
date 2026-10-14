set -x
R=res
ALL="LEACH-CB,DEEC-CB,EECHS-CB,HCB-XL,LEACH-CB+G,EECHS-CB+G,HCB-XL+G,LQ-HCB,LQ-HCB-T,LQ-HCB -R1,LQ-HCB -R2,LQ-HCB -R3,LQ-HCB -FC,LQ-HCB -R1R2R3,LQ-HCB RF-only,LQ-HCB PV-only"
MAIN="LEACH-CB,EECHS-CB,HCB-XL,LEACH-CB+G,EECHS-CB+G,HCB-XL+G,LQ-HCB,LQ-HCB-T,LQ-HCB -FC"
python3 lqhcb.py --scen S1 --seeds 0:30 --protos "$ALL" --out $R/S1.json
python3 lqhcb.py --scen S2 --seeds 0:10 --protos "$MAIN" --out $R/S2.json
python3 lqhcb.py --scen S3 --seeds 0:10 --protos "$MAIN" --out $R/S3.json
for B in 10 40 80; do
 python3 lqhcb.py --scen S1 --seeds 0:10 --protos "LQ-HCB,LQ-HCB-T,EECHS-CB+G,HCB-XL+G,LEACH-CB+G" --over "{\"Bcap\": $B}" --out $R/B$B.json
done
python3 lqhcb.py --scen S1 --seeds 0:10 --protos "$MAIN" --over '{"H": 400}' --out $R/H400.json
for k in 0.25 0.5 2 4; do python3 lqhcb.py --scen S1 --seeds 0:10 --protos "LQ-HCB" --over "{\"kmult\": $k}" --out $R/kap$k.json; done
for K in 1 5 20 50; do python3 lqhcb.py --scen S1 --seeds 0:10 --protos "LQ-HCB" --over "{\"K\": $K}" --out $R/K$K.json; done
for Rc in 20 35; do python3 lqhcb.py --scen S1 --seeds 0:10 --protos "LQ-HCB" --over "{\"Rc\": $Rc}" --out $R/Rc$Rc.json; done
python3 -c "
import lqhcb as m, json
out=[m.run((p,0,'S1',{'trace':1})) for p in ['LQ-HCB','EECHS-CB+G','HCB-XL']]
json.dump(out,open('$R/trace.json','w'))"
echo ALLDONE
