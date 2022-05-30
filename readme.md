<h1>VISUAL ASSISTANCE FOR VISUALLY IMPAIRED: API</h1>

## <div align="center">Documentation</div>


<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/HuynhDoTanThanh/API/blob/master/requirements.txt) in a
[**Python==3.8.**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).
  
```bash
git clone https://github.com/HuynhDoTanThanh/API.git
cd API
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt  # install
```
  
 
<details open>
<summary>Start API</summary>
 
 Get host_ip address and run 
 ```bash
python main.py -H host_ip
#python main.py -H 0.0.0.0
``` 
