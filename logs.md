
(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh  101_prepare_data.py 
downloading tinyshakespeare to data/tinyshakespeare.txt
corpus length: 1115394 characters
vocab size: 65
train tokens: 1003854, val tokens: 111540
wrote train.pt, val.pt, meta.json to data


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 102_train_model.py 
using device: cuda
model parameters: 801,664
step 0: train loss 4.2045, val loss 4.1396
step 250: train loss 3.3153, val loss 3.3421
step 500: train loss 2.8636, val loss 2.8543
step 750: train loss 2.6313, val loss 2.6138
step 1000: train loss 2.4899, val loss 2.4857
step 1250: train loss 2.4531, val loss 2.4430
step 1500: train loss 2.4272, val loss 2.4111
step 1750: train loss 2.3838, val loss 2.3941
step 2000: train loss 2.2922, val loss 2.3010
step 2250: train loss 2.1935, val loss 2.2259
step 2500: train loss 2.1208, val loss 2.1664
step 2750: train loss 2.0479, val loss 2.0983
step 3000: train loss 1.9468, val loss 2.0234
step 3250: train loss 1.8885, val loss 1.9756
step 3500: train loss 1.8378, val loss 1.9555
step 3750: train loss 1.7526, val loss 1.9042
step 4000: train loss 1.7036, val loss 1.8817
step 4250: train loss 1.6852, val loss 1.8517
step 4500: train loss 1.6895, val loss 1.8455
step 4750: train loss 1.6731, val loss 1.8227
step 4999: train loss 1.6174, val loss 1.7920
saved checkpoint to checkpoints/gpt.pt





(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 102_train_model.py 
using device: cuda
model parameters: 801,664
step 0: train loss 4.2045, val loss 4.1396
step 250: train loss 3.3153, val loss 3.3421
step 500: train loss 2.8636, val loss 2.8543
step 750: train loss 2.6313, val loss 2.6138
step 1000: train loss 2.4899, val loss 2.4857
step 1250: train loss 2.4531, val loss 2.4430
step 1500: train loss 2.4272, val loss 2.4111
step 1750: train loss 2.3838, val loss 2.3941
step 2000: train loss 2.2922, val loss 2.3010
step 2250: train loss 2.1935, val loss 2.2259
step 2500: train loss 2.1208, val loss 2.1664
step 2750: train loss 2.0479, val loss 2.0983
step 3000: train loss 1.9468, val loss 2.0234
step 3250: train loss 1.8885, val loss 1.9756
step 3500: train loss 1.8378, val loss 1.9555
step 3750: train loss 1.7526, val loss 1.9042
step 4000: train loss 1.7036, val loss 1.8817
step 4250: train loss 1.6852, val loss 1.8517
step 4500: train loss 1.6895, val loss 1.8455
step 4750: train loss 1.6731, val loss 1.8227
step 5000: train loss 1.5902, val loss 1.8016
step 5250: train loss 1.5701, val loss 1.7798
step 5500: train loss 1.5959, val loss 1.7707
step 5750: train loss 1.5545, val loss 1.7526
step 6000: train loss 1.5293, val loss 1.7377
step 6250: train loss 1.5729, val loss 1.7253
step 6500: train loss 1.5438, val loss 1.7200
step 6750: train loss 1.5098, val loss 1.7058
step 7000: train loss 1.5295, val loss 1.7008
step 7250: train loss 1.4962, val loss 1.6976
step 7500: train loss 1.4945, val loss 1.6824
step 7750: train loss 1.4771, val loss 1.6799
step 8000: train loss 1.4980, val loss 1.6721
step 8250: train loss 1.4903, val loss 1.6619
step 8500: train loss 1.4700, val loss 1.6587
step 8750: train loss 1.4745, val loss 1.6528
step 9000: train loss 1.4334, val loss 1.6351
step 9250: train loss 1.4414, val loss 1.6414
step 9500: train loss 1.4621, val loss 1.6405
step 9750: train loss 1.4316, val loss 1.6372
step 10000: train loss 1.3949, val loss 1.6372
step 10250: train loss 1.4029, val loss 1.6216
step 10500: train loss 1.3952, val loss 1.6228
step 10750: train loss 1.3958, val loss 1.6250
step 11000: train loss 1.3852, val loss 1.6166
step 11250: train loss 1.3614, val loss 1.6144
step 11500: train loss 1.3822, val loss 1.6099
step 11750: train loss 1.3935, val loss 1.6145
step 12000: train loss 1.3898, val loss 1.6081
step 12250: train loss 1.3275, val loss 1.6119
step 12500: train loss 1.3662, val loss 1.5969
step 12750: train loss 1.2992, val loss 1.6031
step 13000: train loss 1.3392, val loss 1.5973
step 13250: train loss 1.3399, val loss 1.5921
step 13500: train loss 1.3374, val loss 1.5971
step 13750: train loss 1.2929, val loss 1.6013
step 14000: train loss 1.3067, val loss 1.5901
step 14250: train loss 1.3271, val loss 1.5982
step 14500: train loss 1.3090, val loss 1.5884
step 14750: train loss 1.2945, val loss 1.6007
step 15000: train loss 1.2910, val loss 1.5913
step 15250: train loss 1.2831, val loss 1.5864
step 15500: train loss 1.2729, val loss 1.5847
step 15750: train loss 1.2838, val loss 1.5965
step 16000: train loss 1.3047, val loss 1.5850
step 16250: train loss 1.3153, val loss 1.5865
step 16500: train loss 1.2566, val loss 1.5862
step 16750: train loss 1.2811, val loss 1.5842
step 17000: train loss 1.2471, val loss 1.5896
step 17250: train loss 1.2761, val loss 1.5911
step 17500: train loss 1.2354, val loss 1.5902
step 17750: train loss 1.2587, val loss 1.5847
step 18000: train loss 1.2671, val loss 1.5924
step 18250: train loss 1.2279, val loss 1.5813
step 18500: train loss 1.2482, val loss 1.5839
step 18750: train loss 1.2432, val loss 1.5895
step 19000: train loss 1.2324, val loss 1.5969
step 19250: train loss 1.2187, val loss 1.5945
step 19500: train loss 1.2228, val loss 1.5841
step 19750: train loss 1.2153, val loss 1.6099
step 20000: train loss 1.2537, val loss 1.6017
step 20250: train loss 1.1847, val loss 1.6088




(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 103_generate_text.py 

Clanged Monter him eastand of his heave
And of and gressond signing her he his would,
And make the brab me in again plent
Sainst a man eyal the kither a liver,
That stand blest you changesshing mine,
The see before so show lords.

Murse:
Shen I songer'd be med from me lay he,
In help her the bloodh in you, murk in theine,
Ther your we is with the the your are no flought.

GLOUCESTER:
Margely; heaves your commant there and on yours,
But I peorps waind make buldied.

HENRY BOLINGBROKE:
My I chole 


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 201_verify_attention.py 
PASS: hand-computed scaled_dot_product_attention
PASS: matches F.scaled_dot_product_attention (non-causal)
PASS: matches F.scaled_dot_product_attention (causal)
PASS: single-head matches nn.MultiheadAttention
PASS: single-head causal-masked matches nn.MultiheadAttention
PASS: make_causal_mask shape/values/broadcast (checked across all batches/heads)
ALL PASS: 201_verify_attention.py

(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 202_verify_layer_norm.py 
PASS: LayerNorm matches nn.LayerNorm (float32)
PASS: LayerNorm matches nn.LayerNorm (float64, tight tolerance)
ALL PASS: 202_verify_layer_norm.py


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 203_verify_multi_head_attention.py 
PASS: forward matches nn.MultiheadAttention (non-causal)
PASS: forward matches nn.MultiheadAttention (causal)
PASS: forward matches nn.MultiheadAttention (float64, tight tolerance)
PASS: backward gradients match (input, in_proj, out_proj)
ALL PASS: 203_verify_multi_head_attention.py


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 204_verify_transformer_block.py 
PASS: shape preserved
PASS: causal property holds (future perturbation does not leak backward)
PASS: no NaN/Inf in forward output or gradients
ALL PASS: 204_verify_transformer_block.py


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 205_verify_gpt.py 
PASS: forward output shape correct
PASS: lm_head and token_embedding weights are tied (same object)
PASS: no NaN/Inf in forward output or gradients
ALL PASS: 205_verify_gpt.py


(.venv) [asingh3450@atl1-1-02-002-29-0 llm_from_sratch]$ ./run.sh 206_verify_overfit.py 
initial loss 2.3081, final loss 0.0590, relative drop 97.44%
ALL PASS: 206_verify_overfit.py