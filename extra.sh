while ! grep -q ALLDONE campaign.log; do sleep 20; done
for Rc in 35 40; do python3 lqhcb.py --scen S3 --seeds 0:10 --protos "LQ-HCB" --over "{\"Rc\": $Rc}" --out res/S3_Rc$Rc.json; done
echo EXTRADONE
