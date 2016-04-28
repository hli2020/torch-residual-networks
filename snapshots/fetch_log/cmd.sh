
# fetch logs from servers

# 1202 layer, fb implementation
sshpass -p "123456" scp -P 2190 hyli@cdcgw.ie.cuhk.edu.hk:/data2/project/fb_resnet/log/2016_04_28_11:38:21_s190_cifar_1202_4gpu.log ./

# stochostic
# n=18 (110)
sshpass -p "123456" scp hyli@192.168.72.149:/media/DATADISK/hyli/project/torch-residual-networks/hyli_stochastic_depth/log/2016_04_28_19:13:54_sto_ls149.log ./

# n =3 (20)
sshpass -p "123456" scp hyli@192.168.72.149:/media/DATADISK/hyli/project/torch-residual-networks/hyli_stochastic_depth/log/2016_04_28_19:17:02_sto_ls149.log ./

# n =9 (56)
sshpass -p "123456" scp hyli@192.168.72.149:/media/DATADISK/hyli/project/torch-residual-networks/hyli_stochastic_depth/log/2016_04_28_19:18:54_sto_ls149.log ./


